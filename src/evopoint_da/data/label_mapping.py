"""Map PDB-derived binding labels onto matching AlphaFold structures."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import Atom, Superimposer
from tqdm import tqdm

from evopoint_da.data.structure import StructureParser


@dataclass
class LabelMappingConfig:
    processed_pdb_dir: Path = Path("data/processed_pdb")
    processed_af2_dir: Path = Path("data/processed_af2")
    raw_pdb_dir: Path = Path("data/raw_pdb")
    raw_af2_dir: Path = Path("data/raw_af2")
    mapping_file: Path = Path("pdb_uniprot_mapping.json")
    output_csv: Path = Path("mapping_report_final.csv")
    debug_limit: int = 5
    min_chain_score: float = 0.15
    max_rmsd: float = 30.0


def create_ca_atoms(coords) -> list[Atom.Atom]:
    atoms = []
    for index, coord in enumerate(coords):
        atoms.append(
            Atom.Atom(
                name="CA",
                coord=coord.numpy() if isinstance(coord, torch.Tensor) else coord,
                bfactor=0,
                occupancy=1,
                altloc=" ",
                fullname=" CA ",
                serial_number=index,
                element="C",
            )
        )
    return atoms


def split_data_by_chain(full_data: dict) -> dict[str, dict]:
    """Split a StructureParser payload into per-chain payloads."""
    if not full_data or "residue_ids" not in full_data:
        return {}

    chains = defaultdict(lambda: {"coords": [], "sequence": [], "residue_ids": [], "labels": []})
    full_sequence = full_data["sequence"]

    for index, residue_id in enumerate(full_data["residue_ids"]):
        chain_id = residue_id.split("_")[0]
        chains[chain_id]["coords"].append(full_data["coords"][index])
        chains[chain_id]["residue_ids"].append(residue_id)
        chains[chain_id]["labels"].append(full_data["labels"][index])
        if index < len(full_sequence):
            chains[chain_id]["sequence"].append(full_sequence[index])

    chain_payloads = {}
    for chain_id, data in chains.items():
        if len(data["sequence"]) < 5:
            continue
        chain_payloads[chain_id] = {
            "coords": np.asarray(data["coords"], dtype=np.float32),
            "sequence": "".join(data["sequence"]),
            "labels": np.asarray(data["labels"], dtype=np.float32),
            "residue_ids": data["residue_ids"],
        }
    return chain_payloads


def find_best_matching_chain(full_pdb_data: dict, af2_sequence: str, *, debug: bool = False):
    """Find the PDB chain with the best sequence coverage against an AF2 sequence."""
    chain_map = split_data_by_chain(full_pdb_data)
    if not chain_map:
        if debug:
            print("   [debug] No usable PDB chains found.")
        return None, None, 0.0

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    aligner.target_end_open_gap_score = 0.0
    aligner.target_end_extend_gap_score = 0.0
    aligner.query_end_open_gap_score = 0.0
    aligner.query_end_extend_gap_score = 0.0

    best_chain_data = None
    best_pdb_sequence = None
    best_score = -1.0
    af2_len = len(af2_sequence)

    for chain_id, data in chain_map.items():
        pdb_sequence = data["sequence"]
        pdb_len = len(pdb_sequence)
        if pdb_len > af2_len * 1.5:
            continue
        try:
            score = aligner.score(af2_sequence, pdb_sequence)
            normalized_score = score / (2 * pdb_len)
        except Exception:
            continue

        if debug:
            print(f"   [debug] Chain {chain_id}: length={pdb_len}, score={normalized_score:.3f}")
        if normalized_score > best_score:
            best_score = normalized_score
            best_chain_data = data
            best_pdb_sequence = pdb_sequence

    return best_chain_data, best_pdb_sequence, best_score


def sequence_based_mapping(
    af2_data: dict,
    pdb_data: dict,
    af2_sequence: str,
    pdb_sequence: str,
    *,
    max_rmsd: float = 30.0,
):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.target_end_open_gap_score = 0.0
    aligner.target_end_extend_gap_score = 0.0
    aligner.query_end_open_gap_score = 0.0
    aligner.query_end_extend_gap_score = 0.0

    try:
        alignment = aligner.align(af2_sequence, pdb_sequence)[0]
    except Exception:
        return None, 999.0, "Alignment error", 0, 0, 0.0

    af2_to_pdb = {}
    af2_indices_rmsd = []
    pdb_indices_rmsd = []
    for (af2_start, af2_end), (pdb_start, pdb_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(af2_end - af2_start):
            af2_index = af2_start + offset
            pdb_index = pdb_start + offset
            if af2_sequence[af2_index] == pdb_sequence[pdb_index]:
                af2_to_pdb[af2_index] = pdb_index
                af2_indices_rmsd.append(af2_index)
                pdb_indices_rmsd.append(pdb_index)

    if len(af2_to_pdb) < 10:
        return None, 999.0, "Too few aligned residues", 0, 0, 0.0

    fixed_coords = af2_data["pos"][af2_indices_rmsd]
    moving_coords = pdb_data["coords"][pdb_indices_rmsd]
    superimposer = Superimposer()
    superimposer.set_atoms(create_ca_atoms(fixed_coords), create_ca_atoms(moving_coords))
    if superimposer.rms > max_rmsd:
        return None, superimposer.rms, f"Extreme RMSD ({superimposer.rms:.1f})", 0, 0, 0.0

    new_labels = torch.zeros(af2_data["pos"].shape[0], dtype=torch.float32)
    pdb_labels = pdb_data["labels"]
    total_pdb_sites = int((pdb_labels > 0.5).sum())
    mapped_count = 0
    site_displacements = []
    rotation, translation = superimposer.rotran
    rotation = torch.from_numpy(rotation).float()
    translation = torch.from_numpy(translation).float()

    for af2_index, pdb_index in af2_to_pdb.items():
        if pdb_index < len(pdb_labels) and pdb_labels[pdb_index] > 0.5:
            new_labels[af2_index] = 1.0
            mapped_count += 1
            pdb_coord = torch.from_numpy(pdb_data["coords"][pdb_index])
            transformed = torch.matmul(pdb_coord, rotation.T) + translation
            site_displacements.append(torch.norm(af2_data["pos"][af2_index] - transformed).item())

    mean_shift = float(np.mean(site_displacements)) if site_displacements else 0.0
    return new_labels, superimposer.rms, "Success", mapped_count, total_pdb_sites, mean_shift


def find_file_robust(directory: Path, lower_pattern: str, upper_pattern: str) -> list[str]:
    matches = glob.glob(str(directory / lower_pattern))
    if matches:
        return matches
    return glob.glob(str(directory / upper_pattern))


def parse_sequence(parser: StructureParser, raw_path: str | Path) -> str | None:
    try:
        parsed = parser.parse_file_with_labels(raw_path)
    except Exception:
        return None
    return parsed["sequence"] if parsed else None


def map_labels(config: LabelMappingConfig) -> dict[str, float | int]:
    if not config.mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {config.mapping_file}")

    parser = StructureParser()
    mapping = json.loads(config.mapping_file.read_text(encoding="utf-8"))
    uniprot_to_pdb = {value.lower(): key.lower() for key, value in mapping.items()}
    af2_files = sorted(config.processed_af2_dir.glob("*.pt"))

    print(f"Scanning {len(af2_files)} processed AF2 files.")
    stats = {
        "matched": 0,
        "failed": 0,
        "skipped": 0,
        "no_chain_found": 0,
        "total_pdb_sites": 0,
        "mapped_af2_sites": 0,
        "site_shifts": [],
    }
    debug_counter = 0

    for af2_path in tqdm(af2_files, desc="Mapping labels"):
        target_pdb_id = None
        current_uniprot = None
        lower_name = af2_path.name.lower()
        for uniprot_id, pdb_id in uniprot_to_pdb.items():
            if uniprot_id in lower_name:
                target_pdb_id = pdb_id
                current_uniprot = uniprot_id
                break
        if not target_pdb_id or not current_uniprot:
            stats["skipped"] += 1
            continue

        raw_af2_files = find_file_robust(
            config.raw_af2_dir,
            f"*{current_uniprot}*.pdb",
            f"*{current_uniprot.upper()}*.pdb",
        )
        raw_pdb_files = find_file_robust(config.raw_pdb_dir, f"{target_pdb_id}.pdb", f"{target_pdb_id.upper()}.pdb")
        if not raw_pdb_files:
            raw_pdb_files = find_file_robust(config.raw_pdb_dir, f"{target_pdb_id}.cif", f"{target_pdb_id.upper()}.cif")
        if not raw_af2_files or not raw_pdb_files:
            stats["skipped"] += 1
            continue

        debug = debug_counter < config.debug_limit
        if debug:
            print(f"\n[debug] {af2_path.name} -> PDB {target_pdb_id}")
            debug_counter += 1

        try:
            af2_data = torch.load(af2_path, weights_only=False)
            af2_sequence = parse_sequence(parser, raw_af2_files[0])
            full_pdb_data = parser.parse_file_with_labels(raw_pdb_files[0], chain_id=None)
            if not af2_sequence or not full_pdb_data:
                stats["failed"] += 1
                continue

            best_chain_data, best_pdb_sequence, score = find_best_matching_chain(
                full_pdb_data,
                af2_sequence,
                debug=debug,
            )
            if not best_chain_data or score < config.min_chain_score:
                stats["no_chain_found"] += 1
                stats["failed"] += 1
                if debug:
                    print(f"   [debug] No matching chain found; best score={score:.3f}")
                continue

            new_labels, rmsd, message, mapped_count, total_sites, mean_shift = sequence_based_mapping(
                af2_data,
                best_chain_data,
                af2_sequence,
                best_pdb_sequence,
                max_rmsd=config.max_rmsd,
            )
            if debug:
                print(f"   [debug] {message}; PDB sites={total_sites}, mapped={mapped_count}, RMSD={rmsd:.2f}")

            if new_labels is None:
                stats["failed"] += 1
                continue

            stats["total_pdb_sites"] += int(total_sites)
            stats["mapped_af2_sites"] += int(mapped_count)
            if mean_shift > 0:
                stats["site_shifts"].append(mean_shift)

            if mapped_count > 0:
                af2_data["y"] = new_labels
                torch.save(af2_data, af2_path)
                stats["matched"] += 1
            else:
                stats["skipped"] += 1
        except Exception as exc:
            stats["failed"] += 1
            if debug:
                print(f"   [debug] Mapping failed: {exc}")

    total = stats["matched"] + stats["failed"] + stats["skipped"] + stats["no_chain_found"]
    success_rate = stats["matched"] / total if total else 0.0
    label_loss = (
        1.0 - (stats["mapped_af2_sites"] / stats["total_pdb_sites"])
        if stats["total_pdb_sites"]
        else 0.0
    )
    average_shift = float(np.mean(stats["site_shifts"])) if stats["site_shifts"] else 0.0

    report = {
        "Total_Samples": total,
        "Success_Rate": success_rate,
        "Sequence_Label_Loss": label_loss,
        "Avg_Site_Shift": average_shift,
        "Mapped_Sites": stats["mapped_af2_sites"],
        "Total_PDB_Sites": stats["total_pdb_sites"],
        "No_Chain_Found": stats["no_chain_found"],
    }
    pd.DataFrame([report]).to_csv(config.output_csv, index=False)

    print("\nLabel mapping completed.")
    print("=" * 50)
    print(f"Alignment success rate : {success_rate:.2%}")
    print(f"Sequence label loss    : {label_loss:.2%}")
    print(f"Average site shift     : {average_shift:.2f} A")
    print(f"Mapped/total sites     : {stats['mapped_af2_sites']}/{stats['total_pdb_sites']}")
    print(f"No matching chain      : {stats['no_chain_found']}")
    print(f"Report saved to        : {config.output_csv}")
    print("=" * 50)

    return report


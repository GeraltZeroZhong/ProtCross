"""Map PDB-derived binding labels onto matching AlphaFold structures."""

from __future__ import annotations

import glob
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import Atom, Superimposer
from tqdm import tqdm

from protcross.data.structure import StructureParser


@dataclass
class LabelMappingConfig:
    processed_pdb_dir: Path = Path("data/processed_pdb")
    processed_af2_dir: Path = Path("data/processed_af2")
    raw_pdb_dir: Path = Path("data/raw_pdb")
    raw_af2_dir: Path = Path("data/raw_af2")
    mapping_file: Path = Path("artifacts/pdb_uniprot_mapping.json")
    output_csv: Path = Path("mapping_report_final.csv")
    debug_limit: int = 5
    min_chain_score: float = 0.15
    max_rmsd: float = 30.0
    in_place: bool = True
    allow_empty_mapping: bool = False
    min_success_rate: float = 0.0


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
            "chain_id": chain_id,
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

    af2_len = int(af2_data["pos"].shape[0])
    pdb_len = int(len(pdb_data["coords"]))
    af2_to_pdb = {}
    af2_indices_rmsd = []
    pdb_indices_rmsd = []
    for (af2_start, af2_end), (pdb_start, pdb_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(af2_end - af2_start):
            af2_index = af2_start + offset
            pdb_index = pdb_start + offset
            if af2_index >= af2_len or pdb_index >= pdb_len:
                continue
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
        if af2_index < len(new_labels) and pdb_index < len(pdb_labels) and pdb_labels[pdb_index] > 0.5:
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


def sequence_from_processed_or_raw(af2_data: dict, raw_sequence: str | None) -> str | None:
    processed_sequence = af2_data.get("sequence")
    if isinstance(processed_sequence, str) and processed_sequence:
        return processed_sequence[: af2_data["pos"].shape[0]]
    return raw_sequence[: af2_data["pos"].shape[0]] if raw_sequence else None


def map_labels(config: LabelMappingConfig) -> dict[str, float | int]:
    if not config.mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {config.mapping_file}")
    for label, directory in (
        ("processed AF2 directory", config.processed_af2_dir),
        ("raw PDB directory", config.raw_pdb_dir),
        ("raw AF2 directory", config.raw_af2_dir),
    ):
        if not directory.exists():
            raise FileNotFoundError(f"{label} not found: {directory}")

    parser = StructureParser()
    mapping = json.loads(config.mapping_file.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError(f"Mapping file must contain a JSON object: {config.mapping_file}")
    uniprot_to_pdb = _reverse_pdb_uniprot_mapping(mapping)
    af2_files = sorted(config.processed_af2_dir.glob("*.pt"))
    if not af2_files:
        raise FileNotFoundError(f"No processed AF2 .pt files found in {config.processed_af2_dir}")

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
        target_pdb_ids = []
        current_uniprot = None
        lower_name = af2_path.name.lower()
        for uniprot_id, pdb_ids in uniprot_to_pdb.items():
            if uniprot_id in lower_name:
                target_pdb_ids = pdb_ids
                current_uniprot = uniprot_id
                break
        if not target_pdb_ids or not current_uniprot:
            stats["skipped"] += 1
            continue

        raw_af2_files = find_file_robust(
            config.raw_af2_dir,
            f"*{current_uniprot}*.pdb",
            f"*{current_uniprot.upper()}*.pdb",
        )
        raw_pdb_files = []
        for target_pdb_id in target_pdb_ids:
            raw_pdb_files.extend(
                find_file_robust(config.raw_pdb_dir, f"{target_pdb_id}.pdb", f"{target_pdb_id.upper()}.pdb")
            )
            raw_pdb_files.extend(
                find_file_robust(config.raw_pdb_dir, f"{target_pdb_id}.cif", f"{target_pdb_id.upper()}.cif")
            )
            raw_pdb_files.extend(
                find_file_robust(config.raw_pdb_dir, f"{target_pdb_id}.mmcif", f"{target_pdb_id.upper()}.mmcif")
            )
        if not raw_af2_files or not raw_pdb_files:
            stats["skipped"] += 1
            continue

        debug = debug_counter < config.debug_limit
        if debug:
            print(f"\n[debug] {af2_path.name} -> PDB candidates {', '.join(target_pdb_ids)}")
            debug_counter += 1

        try:
            af2_data = torch.load(af2_path, weights_only=False)
            af2_sequence = sequence_from_processed_or_raw(af2_data, parse_sequence(parser, raw_af2_files[0]))
            if not af2_sequence:
                stats["failed"] += 1
                continue

            combined_labels = torch.zeros(af2_data["pos"].shape[0], dtype=torch.float32)
            total_sites_sum = 0
            mapped_any = False
            no_chain_for_all = True
            site_shifts = []
            provenance_records = []

            for raw_pdb_file in raw_pdb_files:
                full_pdb_data = parser.parse_file_with_labels(raw_pdb_file, chain_id=None)
                if not full_pdb_data:
                    continue

                best_chain_data, best_pdb_sequence, score = find_best_matching_chain(
                    full_pdb_data,
                    af2_sequence,
                    debug=debug,
                )
                if not best_chain_data or score < config.min_chain_score:
                    if debug:
                        print(f"   [debug] No matching chain for {raw_pdb_file}; best score={score:.3f}")
                    continue

                no_chain_for_all = False
                new_labels, rmsd, message, mapped_count, total_sites, mean_shift = sequence_based_mapping(
                    af2_data,
                    best_chain_data,
                    af2_sequence,
                    best_pdb_sequence,
                    max_rmsd=config.max_rmsd,
                )
                if debug:
                    print(
                        f"   [debug] {Path(raw_pdb_file).name}: {message}; "
                        f"PDB sites={total_sites}, mapped={mapped_count}, RMSD={rmsd:.2f}"
                    )

                if new_labels is None:
                    continue

                total_sites_sum += int(total_sites)
                combined_labels = torch.maximum(combined_labels, new_labels.float())
                if mean_shift > 0:
                    site_shifts.append(mean_shift)
                mapped_any = mapped_any or mapped_count > 0
                provenance_records.append(
                    {
                        "raw_pdb_path": str(raw_pdb_file),
                        "raw_pdb_sha256": _file_sha256(Path(raw_pdb_file)),
                        "matched_chain_id": best_chain_data.get("chain_id"),
                        "chain_score": float(score),
                        "rmsd": float(rmsd),
                        "mapped_sites": int(mapped_count),
                        "total_pdb_sites": int(total_sites),
                        "mean_site_shift": float(mean_shift),
                        "message": message,
                    }
                )

            if no_chain_for_all:
                stats["no_chain_found"] += 1
                stats["failed"] += 1
                continue
            if not mapped_any:
                stats["failed"] += 1
                continue

            mapped_count = int((combined_labels > 0.5).sum())
            stats["total_pdb_sites"] += int(total_sites_sum)
            stats["mapped_af2_sites"] += mapped_count
            stats["site_shifts"].extend(site_shifts)

            if mapped_count > 0:
                af2_data["y"] = combined_labels
                af2_data["label_provenance"] = _label_provenance(
                    config,
                    af2_path=af2_path,
                    raw_af2_files=raw_af2_files,
                    target_pdb_ids=target_pdb_ids,
                    uniprot_id=current_uniprot,
                    records=provenance_records,
                    mapped_count=mapped_count,
                    total_sites=total_sites_sum,
                )
                if config.in_place:
                    tmp_path = af2_path.with_suffix(af2_path.suffix + ".part")
                    torch.save(af2_data, tmp_path)
                    tmp_path.replace(af2_path)
                stats["matched"] += 1
            else:
                stats["skipped"] += 1
        except Exception as exc:
            stats["failed"] += 1
            if debug:
                print(f"   [debug] Mapping failed: {exc}")

    total = stats["matched"] + stats["failed"] + stats["skipped"]
    if total == 0:
        raise RuntimeError("No AF2 samples were scanned for label mapping.")
    success_rate = stats["matched"] / total if total else 0.0
    label_loss = (
        1.0 - (stats["mapped_af2_sites"] / stats["total_pdb_sites"])
        if stats["total_pdb_sites"]
        else 0.0
    )
    average_shift = float(np.mean(stats["site_shifts"])) if stats["site_shifts"] else 0.0

    report = {
        "Total_Samples": total,
        "Matched_Samples": stats["matched"],
        "Failed_Samples": stats["failed"],
        "Skipped_Samples": stats["skipped"],
        "Success_Rate": success_rate,
        "Sequence_Label_Loss": label_loss,
        "Avg_Site_Shift": average_shift,
        "Mapped_Sites": stats["mapped_af2_sites"],
        "Total_PDB_Sites": stats["total_pdb_sites"],
        "No_Chain_Found": stats["no_chain_found"],
    }
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(config.output_csv, index=False)
    _write_mapping_manifest(config, report)

    print("\nLabel mapping completed.")
    print("=" * 50)
    print(f"Alignment success rate : {success_rate:.2%}")
    print(f"Sequence label loss    : {label_loss:.2%}")
    print(f"Average site shift     : {average_shift:.2f} A")
    print(f"Mapped/total sites     : {stats['mapped_af2_sites']}/{stats['total_pdb_sites']}")
    print(f"No matching chain      : {stats['no_chain_found']}")
    print(f"Report saved to        : {config.output_csv}")
    print("=" * 50)

    if not config.allow_empty_mapping:
        if stats["matched"] == 0 or stats["mapped_af2_sites"] == 0:
            raise RuntimeError(
                "Label mapping produced no mapped AF2 labels. "
                "Check PDB-to-UniProt mapping and input directories, or pass --allow-empty-mapping."
            )
        if success_rate < config.min_success_rate:
            raise RuntimeError(
                f"Label mapping success rate {success_rate:.2%} is below required "
                f"{config.min_success_rate:.2%}. Pass --allow-empty-mapping to override."
            )

    return report


def _reverse_pdb_uniprot_mapping(mapping: dict) -> dict[str, list[str]]:
    uniprot_to_pdb: dict[str, list[str]] = defaultdict(list)
    for pdb_id, accessions in mapping.items():
        normalized_pdb_id = str(pdb_id).strip().lower()
        if not normalized_pdb_id:
            continue
        if isinstance(accessions, (list, tuple, set)):
            accession_values = accessions
        else:
            accession_values = [accessions]
        for accession in accession_values:
            normalized_accession = str(accession).strip().lower()
            if normalized_accession and normalized_pdb_id not in uniprot_to_pdb[normalized_accession]:
                uniprot_to_pdb[normalized_accession].append(normalized_pdb_id)
    return dict(uniprot_to_pdb)


def _label_provenance(
    config: LabelMappingConfig,
    *,
    af2_path: Path,
    raw_af2_files: list[str],
    target_pdb_ids: list[str],
    uniprot_id: str,
    records: list[dict],
    mapped_count: int,
    total_sites: int,
) -> dict:
    return {
        "schema_version": "protcross-label-provenance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mapping_file": str(config.mapping_file),
        "mapping_file_sha256": _file_sha256(config.mapping_file),
        "processed_af2_path": str(af2_path),
        "processed_af2_sha256_before_save": _file_sha256(af2_path) if af2_path.exists() else None,
        "raw_af2_files": [
            {"path": str(path), "sha256": _file_sha256(Path(path))}
            for path in raw_af2_files
            if Path(path).exists()
        ],
        "uniprot_id": uniprot_id,
        "target_pdb_ids": target_pdb_ids,
        "min_chain_score": config.min_chain_score,
        "max_rmsd": config.max_rmsd,
        "mapped_sites": int(mapped_count),
        "total_pdb_sites": int(total_sites),
        "records": records,
    }


def _write_mapping_manifest(config: LabelMappingConfig, report: dict) -> None:
    manifest = {
        "schema_version": "protcross-label-mapping-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "processed_pdb_dir": str(config.processed_pdb_dir),
            "processed_af2_dir": str(config.processed_af2_dir),
            "raw_pdb_dir": str(config.raw_pdb_dir),
            "raw_af2_dir": str(config.raw_af2_dir),
            "mapping_file": str(config.mapping_file),
            "mapping_file_sha256": _file_sha256(config.mapping_file),
            "debug_limit": config.debug_limit,
            "min_chain_score": config.min_chain_score,
            "max_rmsd": config.max_rmsd,
            "in_place": config.in_place,
            "allow_empty_mapping": config.allow_empty_mapping,
            "min_success_rate": config.min_success_rate,
        },
        "report": report,
    }
    manifest_path = config.output_csv.with_suffix(config.output_csv.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

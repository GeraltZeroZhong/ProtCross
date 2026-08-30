"""Read-only input quality inspection for ProtCross structure prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings as python_warnings

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Polypeptide import is_aa

from .structure import MAX_ESM_RESIDUES, STANDARD_AA


_KNOWN_MODIFIED_AMINO_ACIDS = frozenset({"MSE", "SEC", "PYL", "SEP", "TPO", "PTR", "CSO", "HYP"})


def inspect_structure(
    file_path: str | Path,
    *,
    chain_id: str | None = None,
    max_len: int = MAX_ESM_RESIDUES,
) -> dict[str, Any]:
    """Describe exactly what ProtCross can score before any model assets are loaded."""
    path = Path(file_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Input structure not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError(f"Unsupported input structure extension: {path.suffix or '<none>'}")
    if max_len <= 0 or max_len > MAX_ESM_RESIDUES:
        raise ValueError(f"max_len must be between 1 and {MAX_ESM_RESIDUES}")

    parser = PDBParser(QUIET=False) if suffix == ".pdb" else MMCIFParser(QUIET=False)
    try:
        with python_warnings.catch_warnings(record=True) as caught_warnings:
            python_warnings.simplefilter("always", PDBConstructionWarning)
            structure = parser.get_structure("protcross_inspection", str(path))
        models = list(structure)
    except Exception as exc:
        raise ValueError(f"Could not parse {path.name} as a {structure_format(path)} structure: {exc}") from exc
    if not models:
        raise ValueError(f"No coordinate models found in {path}")
    parser_warnings = list(
        dict.fromkeys(
            message
            for item in caught_warnings
            if not (message := str(item.message)).startswith("Ignoring unrecognized record")
        )
    )

    model = models[0]
    coordinate_chains = sorted((str(chain.id) for chain in model), key=_chain_sort_key)
    available_chains = sorted(
        (
            str(chain.id)
            for chain in model
            if any(
                residue.id[0] == " "
                and residue.get_resname().strip().upper() in STANDARD_AA
                and "CA" in residue
                for residue in chain
            )
        ),
        key=_chain_sort_key,
    )
    if chain_id is not None and chain_id not in coordinate_chains:
        available = ", ".join(_display_chain(value) for value in available_chains) or "none"
        raise ValueError(f"Chain {chain_id!r} was not found. Available chains: {available}.")

    selected_chains = [chain_id] if chain_id is not None else available_chains
    chain_summaries: list[dict[str, Any]] = []
    modified_names: set[str] = set()
    totals = {
        "scorable": 0,
        "standard_missing_ca": 0,
        "modified_or_nonstandard_amino_acid": 0,
        "alternate_ca": 0,
        "sequence_break": 0,
        "numbering_gap": 0,
    }

    for chain in sorted(model, key=lambda item: _chain_sort_key(str(item.id))):
        if str(chain.id) not in selected_chains:
            continue
        scorable = []
        missing_ca = 0
        nonstandard = 0
        alternate_ca = 0
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            standard_polymer = residue.id[0] == " " and resname in STANDARD_AA
            if standard_polymer:
                if "CA" not in residue:
                    missing_ca += 1
                    continue
                ca = residue["CA"]
                if ca.is_disordered():
                    alternate_ca += 1
                scorable.append(residue)
                continue
            if is_aa(residue, standard=False) or resname in _KNOWN_MODIFIED_AMINO_ACIDS:
                nonstandard += 1
                modified_names.add(resname)

        # Bio.PDB preserves record order, while prediction canonicalizes
        # residues before ESM/PointNet. Keep diagnostics invariant to harmless
        # ATOM block reordering as well.
        scorable.sort(
            key=lambda residue: (
                int(residue.id[1]),
                str(residue.id[2]).strip(),
                residue.get_resname().strip().upper(),
            )
        )

        breaks = []
        numbering_gaps = []
        for previous, current in zip(scorable, scorable[1:]):
            distance = float(np.linalg.norm(previous["CA"].get_coord() - current["CA"].get_coord()))
            if distance > 8.0:
                breaks.append(
                    {
                        "after": _residue_label(previous),
                        "before": _residue_label(current),
                        "ca_distance_angstrom": distance,
                    }
                )
            previous_number = int(previous.id[1])
            current_number = int(current.id[1])
            if current_number - previous_number > 1:
                numbering_gaps.append(
                    {
                        "after": _residue_label(previous),
                        "before": _residue_label(current),
                        "missing_author_numbers": current_number - previous_number - 1,
                    }
                )
        count = len(scorable)
        summary = {
            "chain_id": str(chain.id),
            "scorable_residue_count": count,
            "standard_residues_missing_ca": missing_ca,
            "modified_or_nonstandard_amino_acids": nonstandard,
            "alternate_ca_residues": alternate_ca,
            "sequence_break_count": len(breaks),
            "sequence_breaks": breaks[:20],
            "numbering_gap_count": len(numbering_gaps),
            "numbering_gaps": numbering_gaps[:20],
            "first_scorable_residue": _residue_label(scorable[0]) if scorable else None,
            "last_scorable_residue": _residue_label(scorable[-1]) if scorable else None,
            "exceeds_esm_context": count > max_len,
            "residues_over_context_limit": max(0, count - max_len),
        }
        chain_summaries.append(summary)
        totals["scorable"] += count
        totals["standard_missing_ca"] += missing_ca
        totals["modified_or_nonstandard_amino_acid"] += nonstandard
        totals["alternate_ca"] += alternate_ca
        totals["sequence_break"] += len(breaks)
        totals["numbering_gap"] += len(numbering_gaps)

    if totals["scorable"] == 0:
        scope = f" for chain {chain_id!r}" if chain_id is not None else ""
        raise ValueError(f"No standard amino-acid residues with CA atoms found{scope} in {path}.")

    longest_chain = max((item["scorable_residue_count"] for item in chain_summaries), default=0)
    warnings = _inspection_warnings(
        model_count=len(models),
        selected_chain_count=len([item for item in chain_summaries if item["scorable_residue_count"]]),
        missing_ca=totals["standard_missing_ca"],
        nonstandard=totals["modified_or_nonstandard_amino_acid"],
        modified_names=modified_names,
        alternate_ca=totals["alternate_ca"],
        sequence_breaks=totals["sequence_break"],
        numbering_gaps=totals["numbering_gap"],
        parser_warnings=parser_warnings,
        requires_truncation=longest_chain > max_len,
    )
    return {
        "schema_version": "protcross-structure-inspection-v1",
        "input_structure": str(path),
        "format": structure_format(path),
        "file_size_bytes": path.stat().st_size,
        "file_mtime_ns": path.stat().st_mtime_ns,
        "model_count": len(models),
        "model_scored": str(model.id),
        "coordinate_chains": coordinate_chains,
        "available_chains": available_chains,
        "selected_chains": selected_chains,
        "chain_summaries": chain_summaries,
        "scorable_residue_count": totals["scorable"],
        "standard_residues_missing_ca": totals["standard_missing_ca"],
        "modified_or_nonstandard_amino_acids": totals["modified_or_nonstandard_amino_acid"],
        "modified_residue_names": sorted(modified_names),
        "alternate_ca_residues": totals["alternate_ca"],
        "sequence_break_count": totals["sequence_break"],
        "numbering_gap_count": totals["numbering_gap"],
        "longest_chain_context": longest_chain,
        "max_len": max_len,
        "requires_truncation": longest_chain > max_len,
        "warnings": warnings,
        "parser_warnings": parser_warnings,
        "input_interpretation": {
            "coordinate_model": "first_model_only",
            "assembly": "coordinates_as_supplied; biological assembly operators are not applied",
            "residue_basis": "20 standard ATOM amino-acid residues with a CA atom",
            "sequence_basis": "coordinate-observed scorable residues, independently embedded per chain",
            "geometry_basis": "CA point cloud",
        },
    }


def format_structure_inspection(inspection: dict[str, Any]) -> str:
    """Return a concise, human-readable inspection report."""
    chains = ", ".join(_display_chain(str(value)) for value in inspection["available_chains"]) or "none"
    lines = [
        f"Input: {inspection['input_structure']}",
        f"Format: {inspection['format']}",
        f"Models: {inspection['model_count']} (ProtCross scores the first model)",
        f"Chains: {chains}",
        f"Scorable residues: {inspection['scorable_residue_count']}",
        f"Longest chain context: {inspection['longest_chain_context']} / {inspection['max_len']}",
        "Assembly: supplied coordinates only (biological-assembly operators are not applied)",
    ]
    for chain in inspection["chain_summaries"]:
        lines.append(
            f"  Chain {_display_chain(chain['chain_id'])}: {chain['scorable_residue_count']} scorable; "
            f"{chain['standard_residues_missing_ca']} missing CA; "
            f"{chain['modified_or_nonstandard_amino_acids']} modified/non-standard; "
            f"{chain['sequence_break_count']} coordinate break(s); "
            f"{chain['numbering_gap_count']} numbering gap(s)"
        )
    for warning in inspection["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append("Ready for prediction." if not inspection["requires_truncation"] else "Prediction requires --allow-truncation.")
    return "\n".join(lines)


def structure_format(path: str | Path) -> str:
    return "PDB" if Path(path).suffix.lower() == ".pdb" else "mmCIF"


def _inspection_warnings(
    *,
    model_count: int,
    selected_chain_count: int,
    missing_ca: int,
    nonstandard: int,
    modified_names: set[str],
    alternate_ca: int,
    sequence_breaks: int,
    numbering_gaps: int,
    parser_warnings: list[str],
    requires_truncation: bool,
) -> list[str]:
    warnings = []
    if parser_warnings:
        preview = "; ".join(parser_warnings[:3])
        suffix = f"; and {len(parser_warnings) - 3} more" if len(parser_warnings) > 3 else ""
        warnings.append(f"Structure parser reported: {preview}{suffix}")
    if model_count > 1:
        warnings.append(f"Input contains {model_count} models; only the first coordinate model is scored.")
    if selected_chain_count > 1:
        warnings.append(
            f"All {selected_chain_count} selected chains are used in the supplied coordinate assembly. "
            "ProtCross does not generate a biological assembly; use --chain for a chain-only analysis."
        )
    if missing_ca:
        warnings.append(f"{missing_ca} standard amino-acid residue(s) without CA coordinates are skipped.")
    if nonstandard:
        names = f" ({', '.join(sorted(modified_names))})" if modified_names else ""
        warnings.append(
            f"{nonstandard} modified/non-standard amino-acid residue(s){names} are skipped and absent from the ESM sequence."
        )
    if alternate_ca:
        warnings.append(
            f"{alternate_ca} residue(s) have alternate CA conformations; Bio.PDB's selected conformer supplies model geometry."
        )
    if sequence_breaks:
        warnings.append(
            f"Detected {sequence_breaks} within-chain CA discontinuity/discontinuities (>8 Å). "
            "Coordinate-observed residues on either side are still concatenated for ESM-C."
        )
    if numbering_gaps:
        warnings.append(
            f"Detected {numbering_gaps} author-residue numbering gap(s). Numbering gaps do not prove missing "
            "coordinates, but residue mapping should be checked against the source sequence."
        )
    if requires_truncation:
        warnings.append("At least one chain exceeds the ESM-C context limit and is not scored unless truncation is enabled.")
    return warnings


def _residue_label(residue) -> str:
    insertion_code = str(residue.id[2]).strip()
    return f"{int(residue.id[1])}{insertion_code}"


def _chain_sort_key(chain_id: str) -> tuple[int, str]:
    return (0, chain_id) if chain_id.strip() else (1, "")


def _display_chain(chain_id: str) -> str:
    return chain_id if chain_id.strip() else "<blank>"

"""PDB output helpers for ProtCross predictions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Mapping

import numpy as np
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser

STANDARD_AA = {
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
}


def write_bfactor_pdb(
    input_structure: str | Path,
    output_pdb: str | Path,
    residue_ids: Iterable[str],
    probabilities: Iterable[float] | np.ndarray,
    *,
    missing_value: float | None = None,
    residue_metadata: Iterable[Mapping[str, object]] | None = None,
) -> None:
    """Write residue probabilities into the B-factor column of a structure file.

    By default only scored protein residues are modified. Pass ``missing_value``
    to assign a value to unscored atoms for legacy behavior.
    """
    input_structure = Path(input_structure)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    parser = MMCIFParser(QUIET=True) if input_structure.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("protcross_prediction", str(input_structure))
    score_by_residue_id = {residue_id: float(score) for residue_id, score in zip(residue_ids, probabilities)}
    score_by_residue_key: dict[str, float] = {}
    if residue_metadata is not None:
        for metadata, score in zip(residue_metadata, probabilities):
            residue_key = metadata.get("residue_key")
            if residue_key:
                score_by_residue_key[str(residue_key)] = float(score)
    allow_legacy_fallback = not score_by_residue_key or all(
        key.startswith("legacy:") for key in score_by_residue_key
    )

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_key = _residue_key(model.id, chain.id, residue.id, residue.get_resname())
                legacy_id = _legacy_residue_id(chain.id, residue.id)
                score = score_by_residue_key.get(residue_key)
                if score is None and allow_legacy_fallback and _can_use_legacy_fallback(model.id, residue):
                    score = score_by_residue_id.get(legacy_id)
                if score is None:
                    if missing_value is None:
                        continue
                    score = missing_value
                for atom in residue.get_unpacked_list():
                    atom.set_bfactor(score)

    io = MMCIFIO() if output_pdb.suffix.lower() in {".cif", ".mmcif"} else PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))


def _residue_key(model_id, chain_id: str, residue_id: tuple, resname: str | None = None) -> str:
    hetfield = residue_id[0].strip() or "ATOM"
    insertion_code = residue_id[2].strip()
    resname_part = resname.strip().upper() if resname else ""
    return (
        f"model:{model_id}|chain:{chain_id}|het:{hetfield}|resseq:{int(residue_id[1])}|"
        f"icode:{insertion_code}|resname:{resname_part}"
    )


def _legacy_residue_id(chain_id: str, residue_id: tuple) -> str:
    insertion_code = residue_id[2].strip()
    residue_suffix = f"{int(residue_id[1])}{insertion_code}" if insertion_code else str(int(residue_id[1]))
    return f"{chain_id}_{residue_suffix}"


def _can_use_legacy_fallback(model_id, residue) -> bool:
    return (
        model_id == 0
        and residue.id[0] == " "
        and residue.get_resname().strip().upper() in STANDARD_AA
    )

"""PDB output helpers for ProtCross predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser


def write_bfactor_pdb(
    input_structure: str | Path,
    output_pdb: str | Path,
    residue_ids: Iterable[str],
    probabilities: Iterable[float] | np.ndarray,
    *,
    missing_value: float = 0.0,
) -> None:
    """Write residue probabilities into the B-factor column of a PDB file."""
    input_structure = Path(input_structure)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    parser = MMCIFParser(QUIET=True) if input_structure.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("protcross_prediction", str(input_structure))
    score_map = {residue_id: float(score) for residue_id, score in zip(residue_ids, probabilities)}

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_key = f"{chain.id}_{residue.id[1]}"
                score = score_map.get(residue_key, missing_value)
                for atom in residue:
                    atom.set_bfactor(score)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))


"""Structure parsing utilities used by preprocessing and inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser, Selection
from Bio.SeqUtils import seq1

MAX_ESM_RESIDUES = 1022

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

WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "DOD"})

DEFAULT_IGNORED_HETATM_RESNAMES = frozenset(
    {
        # Common crystallization additives, buffers, and solvents.
        "GOL",
        "EDO",
        "PEG",
        "PE4",
        "PG4",
        "PGE",
        "DMS",
        "ACT",
        "ACY",
        "FMT",
        "MES",
        "TRS",
        # Common salts and phosphate/sulfate-like crystallization species.
        "SO4",
        "PO4",
        "NO3",
        # Monatomic ions that should not define small-molecule binding labels by default.
        "NA",
        "CL",
        "K",
        "MG",
        "CA",
        "ZN",
        "MN",
        "FE",
        "CU",
        "CO",
        "NI",
        "CD",
        "HG",
        "CS",
        "RB",
        "LI",
        "F",
        "BR",
        "I",
        # Protein-like crystallographic artifacts or terminal caps.
        "MSE",
        "ACE",
        "NME",
    }
)


def truncate_parsed_structure(parsed: Dict, max_len: int = MAX_ESM_RESIDUES) -> Dict:
    """Return a shallow copy truncated to the ESM-C residue context limit."""
    if len(parsed["sequence"]) <= max_len:
        return parsed

    out = dict(parsed)
    out["sequence"] = parsed["sequence"][:max_len]
    for key in ("coords", "plddts", "residue_ids", "labels"):
        if key in out:
            out[key] = out[key][:max_len]
    out["truncated"] = True
    out["original_length"] = len(parsed["sequence"])
    return out


class StructureParser:
    """Parse PDB/mmCIF files into residue-level point-cloud inputs."""

    def __init__(
        self,
        *,
        ignored_hetatm_resnames: Iterable[str] | None = None,
        ligand_neighbor_cutoff: float = 6.0,
    ) -> None:
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        ignored = DEFAULT_IGNORED_HETATM_RESNAMES if ignored_hetatm_resnames is None else ignored_hetatm_resnames
        self.ignored_hetatm_resnames = WATER_RESNAMES | frozenset(
            resname.strip().upper() for resname in ignored if resname and resname.strip()
        )
        self.ligand_neighbor_cutoff = float(ligand_neighbor_cutoff)

    def parse_file_with_labels(self, file_path: str | Path, chain_id: Optional[str] = None) -> Optional[Dict]:
        """Parse a structure and infer residue labels from nearby ligand atoms.

        The returned dict preserves the public schema used by the 0.1.0 scripts:
        ``coords``, ``sequence``, ``plddts``, ``residue_ids`` and ``labels``.
        """
        file_path = Path(file_path)
        parser = self.pdb_parser if file_path.suffix.lower() == ".pdb" else self.cif_parser

        try:
            structure = parser.get_structure("protein", str(file_path))
            model = next(iter(structure))
        except Exception:
            return None

        all_atoms = [
            atom
            for atom in Selection.unfold_entities(model, "A")
            if not self._is_ignored_hetatm_residue(atom.get_parent())
        ]
        if not all_atoms:
            return None

        neighbor_search = NeighborSearch(all_atoms)
        coords, seq_chars, plddts, residue_ids, labels = [], [], [], [], []

        for chain in model:
            if chain_id and chain.id != chain_id:
                continue

            for residue in chain:
                res_name = residue.get_resname().strip().upper()
                is_standard = residue.id[0] == " " and res_name in STANDARD_AA
                if not is_standard or "CA" not in residue:
                    continue

                ca_atom = residue["CA"]
                coords.append(ca_atom.get_coord())
                plddts.append(ca_atom.get_bfactor())
                residue_ids.append(f"{chain.id}_{residue.id[1]}")

                try:
                    aa = seq1(res_name)
                except Exception:
                    aa = "X"
                seq_chars.append(aa if len(aa) == 1 else "X")

                labels.append(self._has_ligand_neighbor(neighbor_search, ca_atom, residue))

        if not coords:
            return None

        coords_np = np.asarray(coords, dtype=np.float32)
        coords_np -= coords_np.mean(axis=0)

        return {
            "coords": coords_np,
            "sequence": "".join(seq_chars),
            "plddts": np.asarray(plddts, dtype=np.float32),
            "residue_ids": residue_ids,
            "labels": np.asarray(labels, dtype=np.float32),
            "truncated": False,
            "original_length": len(seq_chars),
        }

    def _is_ignored_hetatm_residue(self, residue) -> bool:
        return residue.get_resname().strip().upper() in self.ignored_hetatm_resnames

    def _has_ligand_neighbor(self, neighbor_search: NeighborSearch, ca_atom, residue) -> float:
        for neighbor_atom in neighbor_search.search(ca_atom.get_coord(), self.ligand_neighbor_cutoff):
            neighbor_residue = neighbor_atom.get_parent()
            if neighbor_residue == residue:
                continue
            if self._is_ignored_hetatm_residue(neighbor_residue):
                continue

            neighbor_name = neighbor_residue.get_resname().strip().upper()
            is_ligand_residue = neighbor_name not in STANDARD_AA or neighbor_residue.id[0] != " "
            if is_ligand_residue:
                return 1.0

        return 0.0

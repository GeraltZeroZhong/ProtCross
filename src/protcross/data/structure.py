"""Structure parsing utilities used by preprocessing and inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser, Selection
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
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
    for key in ("coords", "raw_coords", "plddts", "residue_ids", "residue_metadata", "labels"):
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
        ``coords`` are centered for model input; ``raw_coords`` preserve the
        input coordinate frame for downstream pocket reporting.
        """
        file_path = Path(file_path)
        is_mmcif = file_path.suffix.lower() in {".cif", ".mmcif"}
        parser = self.pdb_parser if file_path.suffix.lower() == ".pdb" else self.cif_parser
        mmcif_ca_metadata = self._load_mmcif_ca_metadata(file_path) if is_mmcif else {}

        try:
            structure = parser.get_structure("protein", str(file_path))
            models = list(structure)
            model = models[0]
        except Exception:
            return None
        structure_warnings = []
        if len(models) > 1:
            structure_warnings.append(
                f"Input contains {len(models)} models; only model {model.id} was parsed and scored."
            )

        all_atoms = [
            atom
            for atom in Selection.unfold_entities(model, "A")
            if not self._is_ignored_hetatm_residue(atom.get_parent())
        ]
        if not all_atoms:
            return None

        neighbor_search = NeighborSearch(all_atoms)
        coords, seq_chars, plddts, residue_ids, residue_metadata, labels = [], [], [], [], [], []

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
                insertion_code = residue.id[2].strip()
                residue_number = int(residue.id[1])
                residue_suffix = f"{residue_number}{insertion_code}" if insertion_code else str(residue_number)
                residue_id = f"{chain.id}_{residue_suffix}"
                residue_ids.append(residue_id)

                try:
                    aa = seq1(res_name)
                except Exception:
                    aa = "X"
                seq_chars.append(aa if len(aa) == 1 else "X")
                metadata = {
                    "residue_id": residue_id,
                    "residue_key": self._residue_key(model.id, chain.id, residue.id, res_name),
                    "residue_id_namespace": "mmcif_auth" if is_mmcif else "pdb",
                    "model_id": str(model.id),
                    "chain_id": chain.id,
                    "auth_asym_id": chain.id,
                    "label_asym_id": None,
                    "residue_number": residue_number,
                    "auth_seq_id": residue_number,
                    "label_seq_id": None,
                    "insertion_code": insertion_code,
                    "resname": res_name,
                    "one_letter_code": aa if len(aa) == 1 else "X",
                    "input_bfactor": float(ca_atom.get_bfactor()),
                }
                if is_mmcif:
                    metadata.update(
                        mmcif_ca_metadata.get(
                            (chain.id, residue_number, insertion_code, res_name),
                            mmcif_ca_metadata.get((chain.id, residue_number, insertion_code, None), {}),
                        )
                    )
                residue_metadata.append(metadata)

                labels.append(self._has_ligand_neighbor(neighbor_search, ca_atom, residue))

        if not coords:
            return None

        raw_coords_np = np.asarray(coords, dtype=np.float32)
        coords_np = raw_coords_np - raw_coords_np.mean(axis=0)

        return {
            "coords": coords_np,
            "raw_coords": raw_coords_np,
            "sequence": "".join(seq_chars),
            "plddts": np.asarray(plddts, dtype=np.float32),
            "residue_ids": residue_ids,
            "residue_metadata": residue_metadata,
            "labels": np.asarray(labels, dtype=np.float32),
            "truncated": False,
            "original_length": len(seq_chars),
            "model_count": len(models),
            "models_scored": [str(model.id)],
            "structure_warnings": structure_warnings,
        }

    @staticmethod
    def _residue_key(model_id, chain_id: str, residue_id: tuple, resname: str | None = None) -> str:
        hetfield = residue_id[0].strip() or "ATOM"
        insertion_code = residue_id[2].strip()
        resname_part = resname.strip().upper() if resname else ""
        return (
            f"model:{model_id}|chain:{chain_id}|het:{hetfield}|resseq:{int(residue_id[1])}|"
            f"icode:{insertion_code}|resname:{resname_part}"
        )

    def _load_mmcif_ca_metadata(self, file_path: Path) -> dict[tuple[str | None, int | None, str, str | None], dict[str, object]]:
        try:
            atom_site = MMCIF2Dict(str(file_path))
        except Exception:
            return {}

        label_atom_ids = self._mmcif_values(atom_site, "_atom_site.label_atom_id")
        auth_atom_ids = self._mmcif_values(atom_site, "_atom_site.auth_atom_id")
        row_count = max(len(label_atom_ids), len(auth_atom_ids))
        if row_count == 0:
            return {}

        def values(field: str) -> list[str | None]:
            return self._mmcif_values(atom_site, field, row_count=row_count)

        group_pdb = values("_atom_site.group_PDB")
        auth_asym_ids = values("_atom_site.auth_asym_id")
        label_asym_ids = values("_atom_site.label_asym_id")
        auth_seq_ids = values("_atom_site.auth_seq_id")
        label_seq_ids = values("_atom_site.label_seq_id")
        insertion_codes = values("_atom_site.pdbx_PDB_ins_code")
        auth_comp_ids = values("_atom_site.auth_comp_id")
        label_comp_ids = values("_atom_site.label_comp_id")
        model_nums = values("_atom_site.pdbx_PDB_model_num")

        metadata_by_key: dict[tuple[str | None, int | None, str, str | None], dict[str, object]] = {}
        for index in range(row_count):
            atom_name = self._clean_mmcif_value(
                label_atom_ids[index] if index < len(label_atom_ids) else None
            ) or self._clean_mmcif_value(auth_atom_ids[index] if index < len(auth_atom_ids) else None)
            if atom_name != "CA":
                continue
            group = self._clean_mmcif_value(group_pdb[index])
            if group and group.upper() != "ATOM":
                continue

            auth_seq_id = self._optional_int(auth_seq_ids[index])
            label_seq_id = self._optional_int(label_seq_ids[index])
            insertion_code = self._clean_mmcif_value(insertion_codes[index]) or ""
            auth_asym_id = self._clean_mmcif_value(auth_asym_ids[index])
            label_asym_id = self._clean_mmcif_value(label_asym_ids[index])
            auth_comp_id = self._clean_mmcif_value(auth_comp_ids[index])
            label_comp_id = self._clean_mmcif_value(label_comp_ids[index])
            model_num = self._clean_mmcif_value(model_nums[index])

            metadata = {
                "residue_id_namespace": "mmcif_auth" if auth_asym_id or auth_seq_id is not None else "mmcif_label",
                "auth_asym_id": auth_asym_id,
                "label_asym_id": label_asym_id,
                "auth_seq_id": auth_seq_id,
                "label_seq_id": label_seq_id,
                "insertion_code": insertion_code,
            }
            if model_num is not None:
                metadata["mmcif_model_num"] = model_num

            for key in (
                (auth_asym_id, auth_seq_id, insertion_code, auth_comp_id),
                (auth_asym_id, auth_seq_id, insertion_code, None),
                (label_asym_id, label_seq_id, insertion_code, label_comp_id),
                (label_asym_id, label_seq_id, insertion_code, None),
            ):
                if key[0] is not None and key[1] is not None:
                    metadata_by_key[key] = metadata
        return metadata_by_key

    @staticmethod
    def _mmcif_values(atom_site: dict, field: str, *, row_count: int | None = None) -> list[str | None]:
        value = atom_site.get(field, [])
        values = value if isinstance(value, list) else [value]
        values = [None if item is None else str(item) for item in values]
        if row_count is not None and len(values) < row_count:
            values.extend([None] * (row_count - len(values)))
        return values

    @staticmethod
    def _clean_mmcif_value(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return None if value in {"", ".", "?"} else value

    @classmethod
    def _optional_int(cls, value: str | None) -> int | None:
        value = cls._clean_mmcif_value(value)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

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

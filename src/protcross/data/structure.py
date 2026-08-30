"""Structure parsing utilities used by preprocessing and inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser, Selection
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionException
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


def parsed_structure_sequence_chunks(parsed: Dict) -> list[tuple[int, int]]:
    """Return contiguous ESM sequence chunks, split by parsed model/chain."""
    sequence = parsed.get("sequence", "")
    metadata = parsed.get("residue_metadata") or []
    if not sequence:
        return []
    if len(metadata) != len(sequence):
        return [(0, len(sequence))]

    chunks: list[tuple[int, int]] = []
    start = 0
    current = _metadata_chain_key(metadata[0])
    for index, item in enumerate(metadata[1:], start=1):
        key = _metadata_chain_key(item)
        if key != current:
            chunks.append((start, index))
            start = index
            current = key
    chunks.append((start, len(sequence)))
    return chunks


def parsed_structure_long_chunks(parsed: Dict, max_len: int = MAX_ESM_RESIDUES) -> list[tuple[int, int]]:
    return [
        (start, end)
        for start, end in parsed_structure_sequence_chunks(parsed)
        if end - start > max_len
    ]


def truncate_parsed_structure_by_chain(parsed: Dict, max_len: int = MAX_ESM_RESIDUES) -> Dict:
    """Truncate each parsed model/chain chunk independently for ESM-C."""
    chunks = parsed_structure_sequence_chunks(parsed)
    if not any(end - start > max_len for start, end in chunks):
        return parsed
    keep_indices: list[int] = []
    for start, end in chunks:
        keep_indices.extend(range(start, min(end, start + max_len)))
    return _subset_parsed_structure(parsed, keep_indices, original_length=len(parsed["sequence"]))


def _subset_parsed_structure(parsed: Dict, indices: list[int], *, original_length: int) -> Dict:
    out = dict(parsed)
    out["sequence"] = "".join(parsed["sequence"][index] for index in indices)
    for key in ("coords", "raw_coords", "plddts", "labels"):
        if key in out:
            out[key] = out[key][indices]
    for key in ("residue_ids", "residue_metadata"):
        if key in out:
            out[key] = [out[key][index] for index in indices]
    out["truncated"] = True
    out["original_length"] = original_length
    return out


def _metadata_chain_key(metadata: dict) -> tuple[str, str]:
    return (str(metadata.get("model_id", "")), str(metadata.get("chain_id", "")))


def canonicalize_parsed_structure(parsed: Dict) -> Dict:
    """Return residue-aligned fields in a stable polymer/identifier order.

    The released PointNet sampling backend is deterministic for a fixed point
    order but is not permutation invariant when neighborhoods are capped.
    Normalizing structure-record order here makes equivalent PDB/mmCIF inputs
    produce the same model input without changing the checkpoint or geometry
    operations used for ordinary chain/residue-ordered files.
    """
    sequence = parsed.get("sequence", "")
    metadata = parsed.get("residue_metadata") or []
    if not sequence or len(metadata) != len(sequence):
        return parsed

    indices = sorted(range(len(metadata)), key=lambda index: (*_canonical_residue_key(metadata[index]), index))
    if indices == list(range(len(indices))):
        return parsed

    out = dict(parsed)
    out["sequence"] = "".join(sequence[index] for index in indices)
    for key in ("coords", "raw_coords", "plddts", "labels"):
        if key in out:
            value = out[key]
            out[key] = value[indices] if isinstance(value, np.ndarray) else [value[index] for index in indices]
    for key in ("residue_ids", "residue_metadata"):
        if key in out:
            out[key] = [out[key][index] for index in indices]
    return out


def _canonical_residue_key(metadata: dict) -> tuple:
    model = _numeric_then_text(metadata.get("model_id"))
    chain_text = str(
        metadata.get("auth_asym_id")
        or metadata.get("chain_id")
        or metadata.get("label_asym_id")
        or ""
    )
    chain = (0, chain_text) if chain_text.strip() else (1, "")
    if metadata.get("label_seq_id") is not None:
        sequence_position = (0, *_numeric_then_text(metadata.get("label_seq_id")))
    elif metadata.get("auth_seq_id") is not None:
        sequence_position = (1, *_numeric_then_text(metadata.get("auth_seq_id")))
    else:
        sequence_position = (2, *_numeric_then_text(metadata.get("residue_number")))
    return (
        model,
        chain,
        sequence_position,
        str(metadata.get("insertion_code") or ""),
        str(metadata.get("label_asym_id") or ""),
        str(metadata.get("resname") or ""),
        str(metadata.get("residue_key") or metadata.get("residue_id") or ""),
    )


def _numeric_then_text(value: object) -> tuple[int, int | str]:
    try:
        return (0, int(str(value)))
    except (TypeError, ValueError):
        return (1, "" if value is None else str(value))


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
        return self._parse_file(file_path, chain_id=chain_id, include_labels=True)

    def parse_file(self, file_path: str | Path, chain_id: Optional[str] = None) -> Optional[Dict]:
        """Parse model inputs without constructing unused ligand labels.

        Prediction uses exactly the same residue selection, canonical ordering,
        coordinates, sequence, and metadata as :meth:`parse_file_with_labels`.
        Skipping the all-atom neighbor index only removes work whose output is
        not consumed by inference.
        """
        return self._parse_file(file_path, chain_id=chain_id, include_labels=False)

    def _parse_file(
        self,
        file_path: str | Path,
        *,
        chain_id: Optional[str],
        include_labels: bool,
    ) -> Optional[Dict]:
        file_path = Path(file_path)
        is_mmcif = file_path.suffix.lower() in {".cif", ".mmcif"}
        parser = self.pdb_parser if file_path.suffix.lower() == ".pdb" else self.cif_parser
        mmcif_ca_metadata = self._load_mmcif_ca_metadata(file_path) if is_mmcif else {}

        try:
            structure = parser.get_structure("protein", str(file_path))
        except (KeyError, ValueError, PDBConstructionException) as exc:
            raise ValueError(f"Could not parse structure {file_path}: {exc}") from exc
        models = list(structure)
        if not models:
            return None
        model = models[0]
        structure_warnings = []
        if len(models) > 1:
            structure_warnings.append(
                f"Input contains {len(models)} models; only model {model.id} was parsed and scored."
            )

        neighbor_search = None
        if include_labels:
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
            if chain_id is not None and chain.id != chain_id:
                continue

            for residue in chain:
                res_name = residue.get_resname().strip().upper()
                is_standard = residue.id[0] == " " and res_name in STANDARD_AA
                if not is_standard or "CA" not in residue:
                    continue

                ca_atom = self._select_atom_conformer(residue["CA"])
                coords.append(ca_atom.get_coord())
                plddts.append(ca_atom.get_bfactor())
                insertion_code = residue.id[2].strip()
                residue_number = int(residue.id[1])
                residue_suffix = f"{residue_number}{insertion_code}" if insertion_code else str(residue_number)
                residue_id = f"{chain.id}_{residue_suffix}"
                residue_ids.append(residue_id)

                aa = seq1(res_name)
                seq_chars.append(aa)
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
                    "one_letter_code": aa,
                    "input_bfactor": float(ca_atom.get_bfactor()),
                }
                if is_mmcif:
                    model_num = self._mmcif_model_num(model)
                    metadata.update(
                        mmcif_ca_metadata.get(
                            (model_num, chain.id, residue_number, insertion_code, res_name),
                            mmcif_ca_metadata.get(
                                (model_num, chain.id, residue_number, insertion_code, None),
                                mmcif_ca_metadata.get(
                                    (None, chain.id, residue_number, insertion_code, res_name),
                                    mmcif_ca_metadata.get((None, chain.id, residue_number, insertion_code, None), {}),
                                ),
                            ),
                        )
                    )
                residue_metadata.append(metadata)

                if neighbor_search is not None:
                    labels.append(self._has_ligand_neighbor(neighbor_search, ca_atom, residue))

        if not coords:
            return None

        raw_coords_np = np.asarray(coords, dtype=np.float32)
        plddts_np = np.asarray(plddts, dtype=np.float32)
        if raw_coords_np.ndim != 2 or raw_coords_np.shape[1] != 3:
            raise ValueError(
                f"Parsed CA coordinates must have shape (N, 3); got {raw_coords_np.shape} for {file_path}."
            )
        if plddts_np.shape != (raw_coords_np.shape[0],):
            raise ValueError(
                f"Parsed CA confidence values must have shape ({raw_coords_np.shape[0]},); "
                f"got {plddts_np.shape} for {file_path}."
            )
        if not np.isfinite(raw_coords_np).all():
            raise ValueError(f"Parsed CA coordinates contain non-finite values: {file_path}.")
        if not np.isfinite(plddts_np).all():
            raise ValueError(f"Parsed CA confidence values contain non-finite values: {file_path}.")
        coords_np = raw_coords_np - raw_coords_np.mean(axis=0)

        parsed = {
            "coords": coords_np,
            "raw_coords": raw_coords_np,
            "sequence": "".join(seq_chars),
            "plddts": plddts_np,
            "residue_ids": residue_ids,
            "residue_metadata": residue_metadata,
            "truncated": False,
            "original_length": len(seq_chars),
            "model_count": len(models),
            "models_scored": [str(model.id)],
            "structure_warnings": structure_warnings,
        }
        if include_labels:
            parsed["labels"] = np.asarray(labels, dtype=np.float32)
        return canonicalize_parsed_structure(parsed)

    @staticmethod
    def _select_atom_conformer(atom):
        """Choose a disordered atom conformer independently of record order."""
        if not atom.is_disordered() or not hasattr(atom, "child_dict"):
            return atom

        candidates = list(atom.child_dict.values())
        if not candidates:
            return atom

        def sort_key(candidate) -> tuple[float, int, str]:
            occupancy = candidate.get_occupancy()
            occupancy_value = float(occupancy) if occupancy is not None else float("-inf")
            altloc = str(candidate.get_altloc() or "").strip()
            preferred_rank = 0 if not altloc else 1 if altloc == "A" else 2
            return (-occupancy_value, preferred_rank, altloc)

        return min(candidates, key=sort_key)

    @staticmethod
    def _residue_key(model_id, chain_id: str, residue_id: tuple, resname: str | None = None) -> str:
        hetfield = residue_id[0].strip() or "ATOM"
        insertion_code = residue_id[2].strip()
        resname_part = resname.strip().upper() if resname else ""
        return (
            f"model:{model_id}|chain:{chain_id}|het:{hetfield}|resseq:{int(residue_id[1])}|"
            f"icode:{insertion_code}|resname:{resname_part}"
        )

    def _load_mmcif_ca_metadata(
        self,
        file_path: Path,
    ) -> dict[tuple[str | None, str | None, int | None, str, str | None], dict[str, object]]:
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

        metadata_by_key: dict[tuple[str | None, str | None, int | None, str, str | None], dict[str, object]] = {}
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
                (model_num, auth_asym_id, auth_seq_id, insertion_code, auth_comp_id),
                (model_num, auth_asym_id, auth_seq_id, insertion_code, None),
                (model_num, label_asym_id, label_seq_id, insertion_code, label_comp_id),
                (model_num, label_asym_id, label_seq_id, insertion_code, None),
            ):
                if key[1] is not None and key[2] is not None:
                    metadata_by_key[key] = metadata
        return metadata_by_key

    @staticmethod
    def _mmcif_model_num(model) -> str:
        serial_num = getattr(model, "serial_num", None)
        if serial_num is not None:
            return str(serial_num)
        model_id = getattr(model, "id", model)
        try:
            return str(int(model_id) + 1)
        except (TypeError, ValueError):
            return str(model_id)

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

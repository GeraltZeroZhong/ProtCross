"""PDB output helpers for ProtCross predictions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Mapping

import numpy as np
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

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
    missing_value: float | None = 0.0,
    residue_metadata: Iterable[Mapping[str, object]] | None = None,
) -> None:
    """Write residue probabilities into the B-factor column of a structure file.

    By default scored protein residues receive ProtCross probabilities and
    unscored atoms receive 0.0, keeping the B-factor column in probability
    units for downstream tools. Pass ``missing_value=None`` to preserve
    original B-factors/pLDDT for unscored atoms.
    """
    input_structure = Path(input_structure)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    input_is_cif = input_structure.suffix.lower() in {".cif", ".mmcif"}
    output_is_cif = output_pdb.suffix.lower() in {".cif", ".mmcif"}
    if input_is_cif and output_is_cif:
        if _write_bfactor_mmcif_preserving_atom_site(
            input_structure,
            output_pdb,
            residue_ids,
            probabilities,
            missing_value=missing_value,
            residue_metadata=residue_metadata,
        ):
            return

    parser = MMCIFParser(QUIET=True) if input_structure.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("protcross_prediction", str(input_structure))
    score_by_residue_id = {residue_id: float(score) for residue_id, score in zip(residue_ids, probabilities)}
    score_by_residue_key: dict[str, float] = {}
    if residue_metadata is not None:
        for metadata, score in zip(residue_metadata, probabilities):
            residue_key = metadata.get("residue_key")
            if residue_key:
                score_by_residue_key[str(residue_key)] = float(score)
    scored_model_ids = _scored_model_ids(score_by_residue_key)
    allow_legacy_fallback = not score_by_residue_key or all(
        key.startswith("legacy:") for key in score_by_residue_key
    )
    if allow_legacy_fallback:
        scored_model_ids.add("0")

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_key = _residue_key(model.id, chain.id, residue.id, residue.get_resname())
                legacy_id = _legacy_residue_id(chain.id, residue.id)
                score = score_by_residue_key.get(residue_key)
                if score is None and allow_legacy_fallback and _can_use_legacy_fallback(model.id, residue):
                    score = score_by_residue_id.get(legacy_id)
                if score is None:
                    if missing_value is None or str(model.id) not in scored_model_ids:
                        continue
                    score = missing_value
                for atom in residue.get_unpacked_list():
                    atom.set_bfactor(score)

    io = MMCIFIO() if output_pdb.suffix.lower() in {".cif", ".mmcif"} else PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))


def _write_bfactor_mmcif_preserving_atom_site(
    input_structure: Path,
    output_cif: Path,
    residue_ids: Iterable[str],
    probabilities: Iterable[float] | np.ndarray,
    *,
    missing_value: float | None,
    residue_metadata: Iterable[Mapping[str, object]] | None,
) -> bool:
    try:
        cif_dict = MMCIF2Dict(str(input_structure))
    except Exception:
        return False
    b_factor_key = "_atom_site.B_iso_or_equiv"
    if b_factor_key not in cif_dict:
        return False

    probabilities_list = [float(score) for score in probabilities]
    residue_ids_list = list(residue_ids)
    score_by_residue_id = dict(zip(residue_ids_list, probabilities_list))
    score_by_atom_site_key: dict[tuple[str | None, str | None, str | None, str, str | None], float] = {}
    scored_models: set[str | None] = set()
    if residue_metadata is not None:
        for metadata, score in zip(residue_metadata, probabilities_list):
            score_value = float(score)
            for key in _metadata_mmcif_keys(metadata):
                score_by_atom_site_key[key] = score_value
            model_num = _metadata_model_num(metadata)
            if model_num is not None:
                scored_models.add(model_num)
    if not score_by_atom_site_key:
        for residue_id, score in score_by_residue_id.items():
            chain_id, _, residue_number = residue_id.partition("_")
            if chain_id and residue_number:
                score_by_atom_site_key[(None, chain_id, residue_number, "", None)] = float(score)
                scored_models.add(None)

    atom_site = {key: _mmcif_values(cif_dict, key) for key in _atom_site_fields(cif_dict)}
    row_count = max((len(values) for values in atom_site.values()), default=0)
    if row_count == 0:
        return False
    for key, values in atom_site.items():
        if len(values) < row_count:
            values.extend(["?"] * (row_count - len(values)))

    b_factors = atom_site[b_factor_key]
    scored_models = scored_models or {"1", None}
    for index in range(row_count):
        model_num = _clean_mmcif_value(_field(atom_site, "_atom_site.pdbx_PDB_model_num", index)) or "1"
        score = _score_for_atom_site_row(atom_site, index, score_by_atom_site_key)
        if score is None:
            if missing_value is None or model_num not in scored_models:
                continue
            score = missing_value
        b_factors[index] = f"{float(score):.6g}"

    patched = dict(cif_dict)
    for key, values in atom_site.items():
        patched[key] = values
    io = MMCIFIO()
    io.set_dict(patched)
    io.save(str(output_cif))
    return True


def _metadata_mmcif_keys(metadata: Mapping[str, object]) -> list[tuple[str | None, str | None, str | None, str, str | None]]:
    model_num = _metadata_model_num(metadata)
    insertion_code = str(metadata.get("insertion_code") or "")
    resname = str(metadata.get("resname") or "") or None
    keys = []
    auth_asym = _optional_str(metadata.get("auth_asym_id") or metadata.get("chain_id"))
    auth_seq = _optional_str(metadata.get("auth_seq_id") or metadata.get("residue_number"))
    label_asym = _optional_str(metadata.get("label_asym_id"))
    label_seq = _optional_str(metadata.get("label_seq_id"))
    if auth_asym and auth_seq:
        keys.append((model_num, auth_asym, auth_seq, insertion_code, resname))
        keys.append((model_num, auth_asym, auth_seq, insertion_code, None))
    if label_asym and label_seq:
        keys.append((model_num, label_asym, label_seq, insertion_code, resname))
        keys.append((model_num, label_asym, label_seq, insertion_code, None))
    return keys


def _metadata_model_num(metadata: Mapping[str, object]) -> str | None:
    if metadata.get("mmcif_model_num") is not None:
        return str(metadata["mmcif_model_num"])
    if metadata.get("model_id") is None:
        return None
    try:
        return str(int(str(metadata["model_id"])) + 1)
    except ValueError:
        return str(metadata["model_id"])


def _score_for_atom_site_row(
    atom_site: Mapping[str, list[str]],
    index: int,
    score_by_atom_site_key: Mapping[tuple[str | None, str | None, str | None, str, str | None], float],
) -> float | None:
    model_num = _clean_mmcif_value(_field(atom_site, "_atom_site.pdbx_PDB_model_num", index)) or "1"
    insertion_code = _clean_mmcif_value(_field(atom_site, "_atom_site.pdbx_PDB_ins_code", index)) or ""
    group_pdb = (_clean_mmcif_value(_field(atom_site, "_atom_site.group_PDB", index)) or "").upper()
    auth_asym = _clean_mmcif_value(_field(atom_site, "_atom_site.auth_asym_id", index))
    auth_seq = _clean_mmcif_value(_field(atom_site, "_atom_site.auth_seq_id", index))
    auth_comp = _clean_mmcif_value(_field(atom_site, "_atom_site.auth_comp_id", index))
    label_asym = _clean_mmcif_value(_field(atom_site, "_atom_site.label_asym_id", index))
    label_seq = _clean_mmcif_value(_field(atom_site, "_atom_site.label_seq_id", index))
    label_comp = _clean_mmcif_value(_field(atom_site, "_atom_site.label_comp_id", index))
    candidates = [
        (model_num, auth_asym, auth_seq, insertion_code, auth_comp),
        (model_num, label_asym, label_seq, insertion_code, label_comp),
    ]
    if group_pdb == "ATOM":
        candidates.extend(
            [
                (model_num, auth_asym, auth_seq, insertion_code, None),
                (model_num, label_asym, label_seq, insertion_code, None),
                (None, auth_asym, auth_seq, insertion_code, None),
                (None, label_asym, label_seq, insertion_code, None),
            ]
        )
    for key in candidates:
        if key in score_by_atom_site_key:
            return score_by_atom_site_key[key]
    return None


def _atom_site_fields(cif_dict: Mapping[str, object]) -> list[str]:
    return [key for key in cif_dict if key.startswith("_atom_site.")]


def _mmcif_values(cif_dict: Mapping[str, object], key: str) -> list[str]:
    value = cif_dict.get(key, [])
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values]


def _field(atom_site: Mapping[str, list[str]], key: str, index: int) -> str | None:
    values = atom_site.get(key)
    if not values or index >= len(values):
        return None
    return values[index]


def _clean_mmcif_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return None if value in {"", ".", "?"} else value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    value_str = str(value)
    return value_str if value_str else None


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


def _scored_model_ids(score_by_residue_key: Mapping[str, float]) -> set[str]:
    model_ids: set[str] = set()
    for key in score_by_residue_key:
        if not key.startswith("model:"):
            continue
        model_part = key.split("|", 1)[0]
        _, _, model_id = model_part.partition(":")
        if model_id:
            model_ids.add(model_id)
    return model_ids

from pathlib import Path

import pytest

from protcross.data.inspection import format_structure_inspection, inspect_structure
from protcross.data.structure import MAX_ESM_RESIDUES, StructureParser


def _pdb_atom(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    residue_number: int,
    x: float,
    *,
    record: str = "ATOM",
    y: float = 0.0,
    z: float = 0.0,
    occupancy: float = 1.0,
    bfactor: float = 20.0,
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:^4} {resname:>3} {chain_id:1}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2}\n"
    )


def _write_multimodel_edge_case_pdb(path: Path) -> None:
    lines = ["HEADER    STRUCTURE INSPECTION REGRESSION\n", "MODEL        1\n"]
    lines.extend(
        [
            _pdb_atom(1, "CA", "ALA", "A", 1, 0.0),
            _pdb_atom(2, "N", "GLY", "A", 2, 3.8, element="N"),
            _pdb_atom(3, "C", "GLY", "A", 2, 4.8),
            _pdb_atom(4, "CA", "MSE", "A", 3, 7.6, record="HETATM"),
            _pdb_atom(5, "CA", "LYS", "A", 4, 20.0),
            _pdb_atom(6, "CA", "SER", "B", 1, 1.0),
            _pdb_atom(7, "C1", "LIG", " ", 901, 2.0, record="HETATM"),
        ]
    )
    lines.extend(
        [
            "ENDMDL\n",
            "MODEL        2\n",
            _pdb_atom(8, "CA", "VAL", "Z", 1, 0.0),
            "ENDMDL\n",
            "END\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def test_inspect_pdb_reports_first_model_chains_and_coordinate_quality(tmp_path: Path):
    input_pdb = tmp_path / "edge-cases.pdb"
    _write_multimodel_edge_case_pdb(input_pdb)

    inspection = inspect_structure(input_pdb)

    assert inspection["format"] == "PDB"
    assert inspection["model_count"] == 2
    assert inspection["model_scored"] == "0"
    assert inspection["coordinate_chains"] == ["A", "B", " "]
    assert inspection["available_chains"] == ["A", "B"]
    assert inspection["selected_chains"] == ["A", "B"]
    assert inspection["scorable_residue_count"] == 3
    assert inspection["standard_residues_missing_ca"] == 1
    assert inspection["modified_or_nonstandard_amino_acids"] == 1
    assert inspection["modified_residue_names"] == ["MSE"]
    assert inspection["sequence_break_count"] == 1
    assert inspection["numbering_gap_count"] == 1

    chains = {summary["chain_id"]: summary for summary in inspection["chain_summaries"]}
    assert set(chains) == {"A", "B"}
    assert chains["A"]["scorable_residue_count"] == 2
    assert chains["A"]["standard_residues_missing_ca"] == 1
    assert chains["A"]["modified_or_nonstandard_amino_acids"] == 1
    assert chains["A"]["sequence_break_count"] == 1
    assert chains["A"]["numbering_gap_count"] == 1
    assert chains["A"]["numbering_gaps"][0]["missing_author_numbers"] == 2
    assert chains["A"]["sequence_breaks"][0]["after"] == "1"
    assert chains["A"]["sequence_breaks"][0]["before"] == "4"
    assert chains["A"]["sequence_breaks"][0]["ca_distance_angstrom"] == pytest.approx(20.0)
    assert chains["B"]["scorable_residue_count"] == 1

    warnings = "\n".join(inspection["warnings"])
    assert "only the first coordinate model is scored" in warnings
    assert "All 2 selected chains" in warnings
    assert "without CA coordinates are skipped" in warnings
    assert "modified/non-standard" in warnings
    assert "MSE" in warnings
    assert "CA discontinuity" in warnings
    assert "author-residue numbering gap" in warnings

    report = format_structure_inspection(inspection)
    assert "Models: 2 (ProtCross scores the first model)" in report
    assert "Chains: A, B" in report
    assert "Ready for prediction." in report


def test_inspect_pdb_explicit_chain_and_invalid_or_ligand_only_chain(tmp_path: Path):
    input_pdb = tmp_path / "chains.pdb"
    _write_multimodel_edge_case_pdb(input_pdb)

    chain_a = inspect_structure(input_pdb, chain_id="A")

    assert chain_a["selected_chains"] == ["A"]
    assert [summary["chain_id"] for summary in chain_a["chain_summaries"]] == ["A"]
    assert chain_a["scorable_residue_count"] == 2
    assert not any("All 2 selected chains" in warning for warning in chain_a["warnings"])

    with pytest.raises(ValueError, match=r"Chain 'Q' was not found") as error:
        inspect_structure(input_pdb, chain_id="Q")
    assert "Available chains: A, B, <blank>" in str(error.value)

    with pytest.raises(ValueError, match=r"No standard amino-acid residues with CA atoms found for chain ' '"):
        inspect_structure(input_pdb, chain_id=" ")


def test_inspect_mmcif_uses_author_chain_ids_and_excludes_ligand_only_chain(tmp_path: Path):
    input_cif = tmp_path / "auth-chains.cif"
    input_cif.write_text(
        """\
data_inspection
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA X 1 1 ? 0.000 0.000 0.000 1.00 10.00 ? 101 ALA A CA 1
ATOM 2 C CA . GLY Y 2 1 ? 3.800 0.000 0.000 1.00 11.00 ? 201 GLY B CA 1
HETATM 3 C C1 . LIG Z 3 . ? 2.000 1.000 0.000 1.00 12.00 ? 901 LIG L C1 1
#
""",
        encoding="utf-8",
    )

    inspection = inspect_structure(input_cif)

    assert inspection["format"] == "mmCIF"
    assert inspection["model_count"] == 1
    assert inspection["coordinate_chains"] == ["A", "B", "L"]
    assert inspection["available_chains"] == ["A", "B"]
    assert inspection["selected_chains"] == ["A", "B"]
    assert inspection["scorable_residue_count"] == 2
    assert [summary["chain_id"] for summary in inspection["chain_summaries"]] == ["A", "B"]


def test_inspect_chain_above_default_esm_context_requires_truncation(tmp_path: Path):
    input_pdb = tmp_path / "long-chain.pdb"
    residue_count = MAX_ESM_RESIDUES + 1
    lines = ["HEADER    LONG CHAIN REGRESSION\n"]
    lines.extend(
        _pdb_atom(serial, "CA", "ALA", "A", serial, (serial - 1) * 3.8)
        for serial in range(1, residue_count + 1)
    )
    lines.extend(["TER\n", "END\n"])
    input_pdb.write_text("".join(lines), encoding="utf-8")

    inspection = inspect_structure(input_pdb)

    assert inspection["max_len"] == MAX_ESM_RESIDUES
    assert inspection["longest_chain_context"] == residue_count
    assert inspection["requires_truncation"] is True
    assert inspection["chain_summaries"][0]["exceeds_esm_context"] is True
    assert inspection["chain_summaries"][0]["residues_over_context_limit"] == 1
    assert inspection["sequence_break_count"] == 0
    assert any("exceeds the ESM-C context limit" in warning for warning in inspection["warnings"])
    assert format_structure_inspection(inspection).endswith("Prediction requires --allow-truncation.")


def test_reordered_structure_records_have_identical_prediction_input_and_diagnostics(tmp_path: Path):
    ordered = tmp_path / "ordered.pdb"
    reordered = tmp_path / "reordered.pdb"
    rows = {
        "A1": _pdb_atom(1, "CA", "ALA", "A", 1, 0.0),
        "A2": _pdb_atom(2, "CA", "GLY", "A", 2, 3.8),
        "B1": _pdb_atom(3, "CA", "LYS", "B", 1, 20.0),
    }
    ordered.write_text(rows["A1"] + rows["A2"] + rows["B1"] + "END\n", encoding="utf-8")
    reordered.write_text(rows["B1"] + rows["A2"] + rows["A1"] + "END\n", encoding="utf-8")

    parser = StructureParser()
    parsed_ordered = parser.parse_file_with_labels(ordered)
    parsed_reordered = parser.parse_file_with_labels(reordered)
    inspection_ordered = inspect_structure(ordered)
    inspection_reordered = inspect_structure(reordered)

    assert parsed_ordered is not None and parsed_reordered is not None
    assert parsed_ordered["sequence"] == parsed_reordered["sequence"] == "AGK"
    assert parsed_ordered["residue_ids"] == parsed_reordered["residue_ids"] == ["A_1", "A_2", "B_1"]
    assert (parsed_ordered["raw_coords"] == parsed_reordered["raw_coords"]).all()
    assert (parsed_ordered["coords"] == parsed_reordered["coords"]).all()
    assert inspection_ordered["chain_summaries"] == inspection_reordered["chain_summaries"]
    assert inspection_ordered["sequence_break_count"] == inspection_reordered["sequence_break_count"] == 0
    assert inspection_ordered["numbering_gap_count"] == inspection_reordered["numbering_gap_count"] == 0

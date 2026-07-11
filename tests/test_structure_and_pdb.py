from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from protcross.data import StructureParser, canonicalize_parsed_structure
from protcross.inference.pdb import write_bfactor_pdb

from conftest import PUBLISH_RAW_PDB, require_file


MINIMAL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.120   9.327  1.00 20.00           C
ATOM      3  C   ALA A   1      13.129  14.520   9.617  1.00 20.00           C
ATOM      4  N   GLY A   2      14.104  14.907   9.947  1.00 20.00           N
ATOM      5  CA  GLY A   2      15.560  14.920   9.827  1.00 20.00           C
ATOM      6  C   GLY A   2      16.129  16.320  9.517  1.00 20.00           C
HETATM    7  C1  LIG A 101      12.800  13.300   9.500  1.00 10.00           C
TER
END
"""

ADDITIVE_AND_LIGAND_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.300   0.000   0.000  1.00 20.00           C
ATOM      4  N   GLY A   2      15.000   0.000   0.000  1.00 20.00           N
ATOM      5  CA  GLY A   2      15.000   0.000   0.000  1.00 20.00           C
ATOM      6  C   GLY A   2      16.300   0.000   0.000  1.00 20.00           C
HETATM    7  C1  GOL A 101       0.500   0.000   0.000  1.00 10.00           C
HETATM    8  C1  LIG A 102      15.500   0.000   0.000  1.00 10.00           C
TER
END
"""


def test_structure_parser_reads_publish_sample():
    parser = StructureParser()
    parsed = parser.parse_file_with_labels(require_file(PUBLISH_RAW_PDB))

    assert parsed is not None
    assert len(parsed["sequence"]) == len(parsed["coords"]) == len(parsed["residue_ids"])
    assert parsed["coords"].shape[1] == 3


def test_structure_parser_ignores_default_hetatm_blacklist(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(ADDITIVE_AND_LIGAND_PDB)

    parsed = StructureParser().parse_file_with_labels(input_pdb)

    assert parsed is not None
    assert parsed["labels"].tolist() == [0.0, 1.0]


def test_structure_parser_preserves_raw_coordinates_for_pocket_reporting(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(ADDITIVE_AND_LIGAND_PDB)

    parsed = StructureParser().parse_file_with_labels(input_pdb)

    assert parsed is not None
    np.testing.assert_allclose(parsed["raw_coords"][0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(parsed["raw_coords"][1], [15.0, 0.0, 0.0])
    np.testing.assert_allclose(parsed["coords"].mean(axis=0), [0.0, 0.0, 0.0])
    assert parsed["residue_metadata"][0]["residue_key"] == "model:0|chain:A|het:ATOM|resseq:1|icode:|resname:ALA"
    assert parsed["residue_metadata"][0]["input_bfactor"] == 20.0
    assert parsed["residue_metadata"][0]["residue_id_namespace"] == "pdb"
    assert parsed["residue_metadata"][0]["label_asym_id"] is None


def test_structure_parser_preserves_insertion_code_metadata(tmp_path: Path):
    input_pdb = tmp_path / "insertion.pdb"
    input_pdb.write_text(
        """\
ATOM      1  N   ALA A  10A      0.000   0.000   0.000  1.00 21.00           N
ATOM      2  CA  ALA A  10A      1.000   0.000   0.000  1.00 21.00           C
ATOM      3  C   ALA A  10A      2.000   0.000   0.000  1.00 21.00           C
TER
END
""",
        encoding="utf-8",
    )

    parsed = StructureParser().parse_file_with_labels(input_pdb)

    assert parsed is not None
    assert parsed["residue_ids"] == ["A_10A"]
    assert parsed["residue_metadata"][0]["insertion_code"] == "A"
    assert parsed["residue_metadata"][0]["residue_key"] == "model:0|chain:A|het:ATOM|resseq:10|icode:A|resname:ALA"


def test_structure_parser_can_disable_extra_hetatm_blacklist(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(ADDITIVE_AND_LIGAND_PDB)

    parsed = StructureParser(ignored_hetatm_resnames=()).parse_file_with_labels(input_pdb)

    assert parsed is not None
    assert parsed["labels"].tolist() == [1.0, 1.0]


def test_write_bfactor_pdb_sets_scores(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "output.pdb"
    input_pdb.write_text(MINIMAL_PDB)

    write_bfactor_pdb(input_pdb, output_pdb, ["A_1", "A_2"], np.array([0.25, 0.75]))

    structure = PDBParser(QUIET=True).get_structure("out", str(output_pdb))
    residues = list(structure.get_residues())
    assert residues[0]["CA"].get_bfactor() == 0.25
    assert residues[1]["CA"].get_bfactor() == 0.75
    assert residues[2]["C1"].get_bfactor() == 0.0


def test_write_bfactor_can_emit_mmcif(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    output_cif = tmp_path / "output.cif"
    input_pdb.write_text(MINIMAL_PDB)

    write_bfactor_pdb(input_pdb, output_cif, ["A_1", "A_2"], np.array([0.25, 0.75]))

    structure = MMCIFParser(QUIET=True).get_structure("out", str(output_cif))
    residues = [residue for residue in structure.get_residues() if residue.id[0] == " "]
    assert residues[0]["CA"].get_bfactor() == 0.25
    assert residues[1]["CA"].get_bfactor() == 0.75


def test_write_bfactor_can_zero_unscored_atoms(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "output.pdb"
    input_pdb.write_text(MINIMAL_PDB)

    write_bfactor_pdb(input_pdb, output_pdb, ["A_1"], np.array([0.25]))

    text = output_pdb.read_text()
    assert "  0.25" in text
    assert "  0.00" in text
    assert " 10.00" not in text


def test_write_bfactor_does_not_score_hetatm_with_colliding_residue_number(tmp_path: Path):
    input_pdb = tmp_path / "collision.pdb"
    output_pdb = tmp_path / "collision.out.pdb"
    input_pdb.write_text(
        """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
HETATM    4  C1  LIG A   1       1.000   1.000   0.000  1.00 10.00           C
TER
END
""",
        encoding="utf-8",
    )

    write_bfactor_pdb(
        input_pdb,
        output_pdb,
        ["A_1"],
        np.array([0.87]),
        residue_metadata=[
            {
                "residue_key": "model:0|chain:A|het:ATOM|resseq:1|icode:|resname:ALA",
            }
        ],
    )

    structure = PDBParser(QUIET=True).get_structure("out", str(output_pdb))
    residues = list(structure.get_residues())
    assert residues[0]["CA"].get_bfactor() == 0.87
    assert residues[1]["C1"].get_bfactor() == 0.0


def test_write_bfactor_legacy_fallback_does_not_score_hetatm_collision(tmp_path: Path):
    input_pdb = tmp_path / "collision.pdb"
    output_pdb = tmp_path / "collision.out.pdb"
    input_pdb.write_text(
        """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
HETATM    4  C1  LIG A   1       1.000   1.000   0.000  1.00 10.00           C
TER
END
""",
        encoding="utf-8",
    )

    write_bfactor_pdb(input_pdb, output_pdb, ["A_1"], np.array([0.87]), residue_metadata=None, missing_value=None)

    structure = PDBParser(QUIET=True).get_structure("out", str(output_pdb))
    residues = list(structure.get_residues())
    assert residues[0]["CA"].get_bfactor() == 0.87
    assert residues[1]["C1"].get_bfactor() == 10.0


def test_write_bfactor_legacy_fallback_only_updates_first_model(tmp_path: Path):
    input_pdb = tmp_path / "models.pdb"
    output_pdb = tmp_path / "models.out.pdb"
    input_pdb.write_text(
        """\
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ENDMDL
MODEL        2
ATOM      3  N   ALA A   1       0.000   0.000   1.000  1.00 50.00           N
ATOM      4  CA  ALA A   1       1.000   0.000   1.000  1.00 50.00           C
ENDMDL
END
""",
        encoding="utf-8",
    )

    write_bfactor_pdb(input_pdb, output_pdb, ["A_1"], np.array([0.42]), residue_metadata=None, missing_value=None)

    models = list(PDBParser(QUIET=True).get_structure("out", str(output_pdb)))
    assert next(models[0].get_residues())["CA"].get_bfactor() == 0.42
    assert next(models[1].get_residues())["CA"].get_bfactor() == 50.0


def test_write_bfactor_updates_all_altloc_atoms_for_scored_residue(tmp_path: Path):
    input_pdb = tmp_path / "altloc.pdb"
    output_pdb = tmp_path / "altloc.out.pdb"
    input_pdb.write_text(
        """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.50 21.00           C
ATOM      3  CA BALA A   1       1.100   0.000   0.000  0.50 22.00           C
ATOM      4  C   ALA A   1       2.000   0.000   0.000  1.00 23.00           C
TER
END
""",
        encoding="utf-8",
    )

    write_bfactor_pdb(input_pdb, output_pdb, ["A_1"], np.array([0.91]), residue_metadata=None)

    text = output_pdb.read_text(encoding="utf-8")
    assert text.count("  0.91") >= 3


def test_write_bfactor_preserves_unscored_model_when_metadata_is_present(tmp_path: Path):
    input_pdb = tmp_path / "models.pdb"
    output_pdb = tmp_path / "models.out.pdb"
    input_pdb.write_text(
        """\
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
ENDMDL
MODEL        2
ATOM      4  N   ALA A   1       0.000   0.000   1.000  1.00 50.00           N
ATOM      5  CA  ALA A   1       1.000   0.000   1.000  1.00 50.00           C
ATOM      6  C   ALA A   1       2.000   0.000   1.000  1.00 50.00           C
ENDMDL
END
""",
        encoding="utf-8",
    )

    write_bfactor_pdb(
        input_pdb,
        output_pdb,
        ["A_1"],
        np.array([0.42]),
        residue_metadata=[
            {
                "residue_key": "model:0|chain:A|het:ATOM|resseq:1|icode:|resname:ALA",
            }
        ],
    )

    structure = PDBParser(QUIET=True).get_structure("out", str(output_pdb))
    models = list(structure)
    assert next(models[0].get_residues())["CA"].get_bfactor() == 0.42
    assert next(models[1].get_residues())["CA"].get_bfactor() == 50.0


def test_structure_parser_reads_mmcif_auth_and_label_metadata(tmp_path: Path):
    input_cif = tmp_path / "input.cif"
    input_cif.write_text(
        """\
data_test
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
ATOM 1 N N . ALA X 1 7 ? 0.000 0.000 0.000 1.00 11.00 ? 101 ALA A N 1
ATOM 2 C CA . ALA X 1 7 ? 1.000 0.000 0.000 1.00 12.00 ? 101 ALA A CA 1
ATOM 3 C C . ALA X 1 7 ? 2.000 0.000 0.000 1.00 13.00 ? 101 ALA A C 1
#
""",
        encoding="utf-8",
    )

    parsed = StructureParser().parse_file_with_labels(input_cif)

    assert parsed is not None
    metadata = parsed["residue_metadata"][0]
    assert metadata["residue_id"] == "A_101"
    assert metadata["residue_id_namespace"] == "mmcif_auth"
    assert metadata["auth_asym_id"] == "A"
    assert metadata["label_asym_id"] == "X"
    assert metadata["auth_seq_id"] == 101
    assert metadata["label_seq_id"] == 7


def test_write_bfactor_mmcif_preserves_auth_and_label_identity(tmp_path: Path):
    input_cif = tmp_path / "input.cif"
    output_cif = tmp_path / "output.cif"
    input_cif.write_text(
        """\
data_test
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
ATOM 1 N N . ALA X 1 7 ? 0.000 0.000 0.000 1.00 11.00 ? 101 ALA A N 1
ATOM 2 C CA . ALA X 1 7 ? 1.000 0.000 0.000 1.00 12.00 ? 101 ALA A CA 1
ATOM 3 C C . ALA X 1 7 ? 2.000 0.000 0.000 1.00 13.00 ? 101 ALA A C 1
#
""",
        encoding="utf-8",
    )
    parsed = StructureParser().parse_file_with_labels(input_cif)

    write_bfactor_pdb(
        input_cif,
        output_cif,
        parsed["residue_ids"],
        np.array([0.66]),
        residue_metadata=parsed["residue_metadata"],
    )

    out = MMCIF2Dict(str(output_cif))
    assert out["_atom_site.label_asym_id"] == ["X", "X", "X"]
    assert out["_atom_site.label_seq_id"] == ["7", "7", "7"]
    assert out["_atom_site.auth_asym_id"] == ["A", "A", "A"]
    assert out["_atom_site.auth_seq_id"] == ["101", "101", "101"]
    assert [float(value) for value in out["_atom_site.B_iso_or_equiv"]] == [0.66, 0.66, 0.66]


def test_write_bfactor_mmcif_preserves_unscored_model_identity(tmp_path: Path):
    input_cif = tmp_path / "models.cif"
    output_cif = tmp_path / "models.out.cif"
    input_cif.write_text(
        """\
data_test
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
ATOM 1 N N . ALA X 1 7 ? 0.000 0.000 0.000 1.00 11.00 ? 101 ALA A N 1
ATOM 2 C CA . ALA X 1 7 ? 1.000 0.000 0.000 1.00 12.00 ? 101 ALA A CA 1
ATOM 3 C C . ALA X 1 7 ? 2.000 0.000 0.000 1.00 13.00 ? 101 ALA A C 1
ATOM 4 N N . ALA X 1 7 ? 0.000 0.000 1.000 1.00 41.00 ? 101 ALA A N 2
ATOM 5 C CA . ALA X 1 7 ? 1.000 0.000 1.000 1.00 42.00 ? 101 ALA A CA 2
ATOM 6 C C . ALA X 1 7 ? 2.000 0.000 1.000 1.00 43.00 ? 101 ALA A C 2
#
""",
        encoding="utf-8",
    )
    parsed = StructureParser().parse_file_with_labels(input_cif)

    assert parsed is not None
    assert parsed["residue_metadata"][0]["mmcif_model_num"] == "1"

    write_bfactor_pdb(
        input_cif,
        output_cif,
        parsed["residue_ids"],
        np.array([0.66]),
        residue_metadata=parsed["residue_metadata"],
    )

    out = MMCIF2Dict(str(output_cif))
    assert [float(value) for value in out["_atom_site.B_iso_or_equiv"]] == [
        0.66,
        0.66,
        0.66,
        41.0,
        42.0,
        43.0,
    ]


def test_structure_parser_records_multimodel_warning(tmp_path: Path):
    input_pdb = tmp_path / "models.pdb"
    input_pdb.write_text(
        """\
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ENDMDL
MODEL        2
ATOM      3  N   ALA A   1       0.000   0.000   1.000  1.00 50.00           N
ATOM      4  CA  ALA A   1       1.000   0.000   1.000  1.00 50.00           C
ENDMDL
END
""",
        encoding="utf-8",
    )

    parsed = StructureParser().parse_file_with_labels(input_pdb)

    assert parsed is not None
    assert parsed["model_count"] == 2
    assert parsed["models_scored"] == ["0"]
    assert "only model 0" in parsed["structure_warnings"][0]


def test_canonicalize_parsed_structure_reorders_every_residue_field_together():
    metadata = [
        {"model_id": "0", "chain_id": "B", "auth_asym_id": "B", "auth_seq_id": 2, "residue_id": "B_2"},
        {"model_id": "0", "chain_id": "A", "auth_asym_id": "A", "auth_seq_id": 2, "residue_id": "A_2"},
        {"model_id": "0", "chain_id": "A", "auth_asym_id": "A", "auth_seq_id": 1, "residue_id": "A_1"},
    ]
    parsed = {
        "sequence": "KGA",
        "coords": np.asarray([[3, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=np.float32),
        "raw_coords": np.asarray([[30, 0, 0], [20, 0, 0], [10, 0, 0]], dtype=np.float32),
        "plddts": np.asarray([30, 20, 10], dtype=np.float32),
        "labels": np.asarray([0, 1, 0], dtype=np.float32),
        "residue_ids": ["B_2", "A_2", "A_1"],
        "residue_metadata": metadata,
    }

    canonical = canonicalize_parsed_structure(parsed)

    assert canonical["sequence"] == "AGK"
    assert canonical["residue_ids"] == ["A_1", "A_2", "B_2"]
    assert canonical["coords"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert canonical["raw_coords"][:, 0].tolist() == [10.0, 20.0, 30.0]
    assert canonical["plddts"].tolist() == [10.0, 20.0, 30.0]
    assert canonical["labels"].tolist() == [0.0, 1.0, 0.0]
    assert parsed["sequence"] == "KGA"


def test_mmcif_nondefault_model_number_is_preserved_when_writing_scores(tmp_path: Path):
    input_cif = tmp_path / "model-five.cif"
    output_cif = tmp_path / "model-five.out.cif"
    input_cif.write_text(
        """\
data_test
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
ATOM 1 N N . ALA X 1 1 ? 0.000 0.000 0.000 1.00 41.00 ? 1 ALA A N 5
ATOM 2 C CA . ALA X 1 1 ? 1.000 0.000 0.000 1.00 42.00 ? 1 ALA A CA 5
ATOM 3 C C . ALA X 1 1 ? 2.000 0.000 0.000 1.00 43.00 ? 1 ALA A C 5
#
""",
        encoding="utf-8",
    )

    parsed = StructureParser().parse_file_with_labels(input_cif)

    assert parsed is not None
    assert parsed["residue_metadata"][0]["mmcif_model_num"] == "5"
    write_bfactor_pdb(
        input_cif,
        output_cif,
        parsed["residue_ids"],
        np.asarray([0.876]),
        residue_metadata=parsed["residue_metadata"],
    )
    out = MMCIF2Dict(str(output_cif))
    assert [float(value) for value in out["_atom_site.B_iso_or_equiv"]] == [0.876, 0.876, 0.876]

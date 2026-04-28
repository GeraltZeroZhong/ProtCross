from pathlib import Path

import numpy as np

from evopoint_da.data import StructureParser
from evopoint_da.inference.pdb import write_bfactor_pdb

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

    text = output_pdb.read_text()
    assert "  0.25" in text
    assert "  0.75" in text

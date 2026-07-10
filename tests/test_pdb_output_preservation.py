from pathlib import Path

import numpy as np

from protcross.inference.pdb import write_bfactor_pdb


def _atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    residue_number: int,
    x: float,
    bfactor: float,
    *,
    record: str = "ATOM",
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:^4} {resname:>3} {chain_id:1}{residue_number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{bfactor:6.2f}          {element:>2}\n"
    )


def test_pdb_annotation_preserves_records_order_and_only_patches_bfactor(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "annotated.pdb"
    input_lines = [
        "HEADER    TRANSFERASE                             10-JUL-26   TST1\n",
        "REMARK   1 ORIGINAL EXPERIMENTAL ANNOTATION\n",
        "CRYST1   44.100   52.200   63.300  90.00  90.00 120.00 P 31 2 1      6\n",
        "LINK         CA  ALA A   1                 C1  LIG A 101     1555   1555  1.80\n",
        "MODEL        1\n",
        _atom_line(1, "N", "ALA", "A", 1, 0.0, 12.34, element="N"),
        "ANISOU    1  N   ALA A   1     1200   1300   1400      0      0      0       N\n",
        _atom_line(2, "CA", "ALA", "A", 1, 1.0, 23.45),
        _atom_line(3, "C1", "LIG", "A", 101, 2.0, 34.56, record="HETATM"),
        "TER       4      ALA A   1\n",
        "ENDMDL\n",
        "CONECT    2    3\n",
        "END\n",
    ]
    input_pdb.write_text("".join(input_lines), encoding="utf-8")

    write_bfactor_pdb(
        input_pdb,
        output_pdb,
        ["A_1"],
        np.array([0.876]),
        residue_metadata=[
            {
                "residue_key": "model:0|chain:A|het:ATOM|resseq:1|icode:|resname:ALA",
                "model_id": "0",
                "chain_id": "A",
                "auth_asym_id": "A",
                "residue_number": 1,
                "auth_seq_id": 1,
                "insertion_code": "",
                "resname": "ALA",
            }
        ],
    )

    output_lines = output_pdb.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(output_lines) == len(input_lines)
    assert [line[:6].strip() for line in output_lines] == [line[:6].strip() for line in input_lines]

    for original, annotated in zip(input_lines, output_lines):
        if original.startswith(("ATOM  ", "HETATM")):
            assert annotated[:60] == original[:60]
            assert annotated[66:] == original[66:]
        else:
            assert annotated == original

    atom_lines = [line for line in output_lines if line.startswith(("ATOM  ", "HETATM"))]
    assert [line[60:66] for line in atom_lines] == ["  0.88", "  0.88", "  0.00"]
    assert "0.876" not in output_pdb.read_text(encoding="utf-8")

    for record in ("HEADER", "REMARK", "CRYST1", "LINK", "CONECT"):
        original = next(line for line in input_lines if line.startswith(record))
        assert original in output_lines


def test_pdb_annotation_preserves_crlf_newlines(tmp_path: Path):
    input_pdb = tmp_path / "windows-input.pdb"
    output_pdb = tmp_path / "windows-output.pdb"
    original = ("HEADER    CRLF TEST\n" + _atom_line(1, "CA", "ALA", "A", 1, 0.0, 20.0) + "END\n").replace(
        "\n", "\r\n"
    )
    input_pdb.write_bytes(original.encode("ascii"))

    write_bfactor_pdb(input_pdb, output_pdb, ["A_1"], np.array([0.5]))

    result = output_pdb.read_bytes()
    assert b"\r\r\n" not in result
    assert result.count(b"\r\n") == original.count("\r\n")
    assert result.split(b"\r\n")[0] == b"HEADER    CRLF TEST"
    assert result.split(b"\r\n")[1][60:66] == b"  0.50"

import numpy as np
import pytest
import torch

from protcross.data.label_mapping import (
    _index_af2_files,
    _reverse_pdb_uniprot_mapping,
    load_processed_pdb_labels,
    sequence_based_mapping,
    sequence_from_processed_or_raw,
    split_data_by_chain,
)


def test_split_data_by_chain_keeps_chain_specific_sequence():
    payload = {
        "coords": [[i, 0, 0] for i in range(10)],
        "sequence": "ACDEFGHIKL",
        "residue_ids": ["A_1", "A_2", "A_3", "A_4", "A_5", "B_1", "B_2", "B_3", "B_4", "B_5"],
        "labels": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    }

    chains = split_data_by_chain(payload)

    assert chains["A"]["sequence"] == "ACDEF"
    assert chains["B"]["sequence"] == "GHIKL"


def test_split_data_by_chain_uses_metadata_for_chain_ids_with_underscores():
    payload = {
        "coords": [[i, 0, 0] for i in range(5)],
        "sequence": "ACDEF",
        "residue_ids": [f"AUTH_A_{index}" for index in range(1, 6)],
        "residue_metadata": [{"auth_asym_id": "AUTH_A"} for _ in range(5)],
        "labels": [0, 1, 0, 0, 1],
    }

    chains = split_data_by_chain(payload)

    assert list(chains) == ["AUTH_A"]
    assert chains["AUTH_A"]["sequence"] == "ACDEF"


def test_reverse_mapping_keeps_multiple_pdbs_for_one_uniprot():
    mapping = {"1abc": "P12345", "2def": "P12345", "3ghi": ["Q9XYZ1", "P12345"]}

    reversed_mapping = _reverse_pdb_uniprot_mapping(mapping)

    assert reversed_mapping["p12345"] == ["1abc", "2def", "3ghi"]
    assert reversed_mapping["q9xyz1"] == ["3ghi"]


def test_sequence_based_mapping_ignores_indices_beyond_truncated_af2_payload():
    coords = np.asarray([[float(i), float(i % 3), float((i * i) % 5)] for i in range(15)], dtype=np.float32)
    af2_data = {"pos": torch.from_numpy(coords[:12])}
    pdb_data = {
        "coords": coords,
        "labels": np.asarray([0] * 10 + [1, 0, 0, 0, 1], dtype=np.float32),
    }

    labels, rmsd, message, mapped_count, total_sites, mean_shift = sequence_based_mapping(
        af2_data,
        pdb_data,
        "A" * 15,
        "A" * 15,
    )

    assert labels is not None
    assert labels.shape[0] == 12
    assert mapped_count == 1
    assert int(labels.sum()) == 1
    assert total_sites == 2
    assert rmsd < 1e-4
    assert message == "Success"
    assert mean_shift < 1e-4


def test_sequence_based_mapping_uses_biopython_row_vector_rotation():
    moving = np.asarray([[i, i % 4, (i * i) % 7] for i in range(12)], dtype=np.float32)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    fixed = moving @ rotation + np.asarray([11.0, -4.0, 3.5], dtype=np.float32)
    labels, rmsd, message, mapped_count, _, mean_shift = sequence_based_mapping(
        {"pos": torch.from_numpy(fixed)},
        {"coords": moving, "labels": np.asarray([1] + [0] * 11, dtype=np.float32)},
        "A" * 12,
        "A" * 12,
    )

    assert labels is not None
    assert message == "Success"
    assert mapped_count == 1
    assert rmsd < 1e-4
    assert mean_shift < 1e-4


def test_af2_filename_matching_is_exact_for_isoforms(tmp_path):
    canonical = tmp_path / "AF-P12345.pt"
    isoform = tmp_path / "AF-P12345-2.pt"
    index = _index_af2_files(
        [canonical, isoform],
        {"p12345": ["1abc"], "p12345-2": ["2def"]},
        "processed AF2",
    )

    assert index == {canonical: "p12345", isoform: "p12345-2"}


def test_af2_filename_matching_rejects_duplicate_accession_inputs(tmp_path):
    with pytest.raises(RuntimeError, match="Ambiguous processed AF2 inputs"):
        _index_af2_files(
            [tmp_path / "AF-P12345.pt", tmp_path / "AF-P12345-F1-model_v6.pt"],
            {"p12345": ["1abc"]},
            "processed AF2",
        )


def test_sequence_from_processed_or_raw_prefers_truncated_processed_sequence():
    af2_data = {"pos": torch.zeros((4, 3)), "sequence": "ACDE"}

    assert sequence_from_processed_or_raw(af2_data, "ACDEFGHIK") == "ACDE"
    assert sequence_from_processed_or_raw({"pos": torch.zeros((4, 3))}, "ACDEFGHIK") == "ACDE"


def test_load_processed_pdb_labels_uses_processed_payload_and_provenance(tmp_path):
    raw = tmp_path / "raw_pdb" / "1abc.pdb"
    raw.parent.mkdir()
    raw.write_text("HEADER raw\n", encoding="utf-8")
    processed_dir = tmp_path / "processed_pdb"
    processed_dir.mkdir()
    processed = processed_dir / "1abc.pt"
    from protcross.data.label_mapping import _file_sha256

    torch.save(
        {
            "pos": torch.ones((5, 3)),
            "y": torch.tensor([0, 1, 0, 0, 1], dtype=torch.float32),
            "sequence": "ACDEF",
            "residue_ids": ["A_1", "A_2", "A_3", "A_4", "A_5"],
            "source_sha256": _file_sha256(raw),
        },
        processed,
    )

    payload = load_processed_pdb_labels(processed_dir, raw)

    assert payload is not None
    assert payload["sequence"] == "ACDEF"
    assert payload["labels"].tolist() == [0, 1, 0, 0, 1]
    assert payload["processed_pdb_path"] == str(processed)
    assert payload["processed_pdb_sha256"] == _file_sha256(processed)


def test_load_processed_pdb_labels_rejects_stale_source(tmp_path):
    raw = tmp_path / "raw_pdb" / "1abc.pdb"
    raw.parent.mkdir()
    raw.write_text("HEADER old\n", encoding="utf-8")
    processed_dir = tmp_path / "processed_pdb"
    processed_dir.mkdir()
    from protcross.data.label_mapping import _file_sha256

    torch.save(
        {
            "pos": torch.ones((5, 3)),
            "y": torch.zeros(5),
            "sequence": "ACDEF",
            "source_sha256": _file_sha256(raw),
        },
        processed_dir / "1abc.pt",
    )
    raw.write_text("HEADER changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        load_processed_pdb_labels(processed_dir, raw)

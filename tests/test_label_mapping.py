import numpy as np
import torch

from protcross.data.label_mapping import (
    _reverse_pdb_uniprot_mapping,
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


def test_sequence_from_processed_or_raw_prefers_truncated_processed_sequence():
    af2_data = {"pos": torch.zeros((4, 3)), "sequence": "ACDE"}

    assert sequence_from_processed_or_raw(af2_data, "ACDEFGHIK") == "ACDE"
    assert sequence_from_processed_or_raw({"pos": torch.zeros((4, 3))}, "ACDEFGHIK") == "ACDE"

from evopoint_da.data.label_mapping import split_data_by_chain


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

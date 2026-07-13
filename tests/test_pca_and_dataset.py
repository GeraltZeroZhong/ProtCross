import json
import torch
import time
from pathlib import Path

import numpy as np
import pytest

from protcross.data import PCAReducer
from protcross.data.dataset import EvoPointDataset


def test_pca_reducer_round_trip(tmp_path):
    reducer = PCAReducer(n_components=2)
    reducer.fit([torch.randn(5, 4), torch.randn(5, 4)])
    path = tmp_path / "pca.pkl"
    reducer.save(path)

    loaded = PCAReducer(n_components=2)
    loaded.load(path)
    transformed = loaded.transform(torch.randn(3, 4))

    assert transformed.shape == (3, 2)


def test_pca_reducer_save_creates_parent_directory(tmp_path):
    reducer = PCAReducer(n_components=2)
    reducer.fit([torch.randn(5, 4), torch.randn(5, 4)])
    path = tmp_path / "artifacts" / "pca.pkl"

    reducer.save(path)

    assert path.exists()


def test_pca_reducer_configures_deterministic_random_state():
    reducer = PCAReducer(n_components=2, random_state=137)

    assert reducer.random_state == 137
    assert reducer.pca.random_state == 137


def test_preprocess_passes_config_seed_to_pca_reducer(tmp_path, monkeypatch):
    import protcross.data.preprocess as preprocess_module

    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    data_dir.mkdir()
    (data_dir / "sample.pdb").write_text("END\n", encoding="utf-8")
    esm_weights = tmp_path / "esm.pth"
    pca_path = tmp_path / "pca.pkl"
    esm_weights.write_bytes(b"esm")
    pca_path.write_bytes(b"pca")
    captured = {}

    class CapturingReducer:
        def __init__(self, n_components, random_state=None):
            captured["n_components"] = n_components
            captured["random_state"] = random_state

        def load(self, path):
            captured["loaded"] = path

    monkeypatch.setattr(preprocess_module, "PCAReducer", CapturingReducer)
    monkeypatch.setattr(preprocess_module, "ESMFeatureExtractor", lambda **kwargs: object())
    monkeypatch.setattr(preprocess_module, "StructureParser", lambda: object())
    monkeypatch.setattr(preprocess_module, "_process_files", lambda *args: 1)

    result = preprocess_module.preprocess_directory(
        preprocess_module.PreprocessConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=esm_weights,
            pca_model_path=pca_path,
            pca_dim=16,
            seed=2718,
            accept_esm_license=True,
        )
    )

    assert result == 1
    assert captured == {
        "n_components": 16,
        "random_state": 2718,
        "loaded": pca_path,
    }


def test_preprocess_failure_preserves_previous_output_and_writes_incomplete_manifest(tmp_path):
    from protcross.data.preprocess import PREPROCESS_MANIFEST, PreprocessConfig, _process_files

    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    data_dir.mkdir()
    output_dir.mkdir()
    good_path = data_dir / "a.pdb"
    failed_path = data_dir / "b.pdb"
    good_path.write_text("good", encoding="utf-8")
    failed_path.write_text("failed", encoding="utf-8")
    previous_output = output_dir / "b.pt"
    previous_output.write_bytes(b"previous-valid-output")

    class Parser:
        def parse_file_with_labels(self, path):
            if path == failed_path:
                raise RuntimeError("simulated parse failure")
            return {
                "coords": np.zeros((1, 3), dtype=np.float32),
                "sequence": "A",
                "plddts": np.array([90.0], dtype=np.float32),
                "residue_ids": ["A_1"],
                "labels": np.array([1.0], dtype=np.float32),
            }

    class Extractor:
        def extract_residue_embeddings(self, sequence):
            return torch.ones((len(sequence), 4), dtype=torch.float32)

    class Reducer:
        def transform(self, embeddings):
            return embeddings

    config = PreprocessConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=tmp_path / "unused-esm.pth",
        fail_on_error=True,
    )

    with pytest.raises(RuntimeError, match="did not complete cleanly"):
        _process_files(config, [good_path, failed_path], Parser(), Extractor(), Reducer())

    manifest = json.loads((output_dir / PREPROCESS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["produced_outputs"] == ["a.pt"]
    assert manifest["failures"][0]["path"] == str(failed_path)
    assert previous_output.read_bytes() == b"previous-valid-output"
    assert (output_dir / "a.pt").exists()
    assert not list(output_dir.glob("*.part"))
    assert not list(output_dir.glob(".*.part"))


def test_preprocess_clean_run_writes_complete_manifest(tmp_path):
    from protcross.data.preprocess import PREPROCESS_MANIFEST, PreprocessConfig, _process_files

    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    data_dir.mkdir()
    output_dir.mkdir()
    input_path = data_dir / "sample.pdb"
    input_path.write_text("sample", encoding="utf-8")

    class Parser:
        def parse_file_with_labels(self, path):
            return {
                "coords": np.zeros((1, 3), dtype=np.float32),
                "sequence": "A",
                "plddts": np.array([90.0], dtype=np.float32),
                "residue_ids": ["A_1"],
                "labels": np.array([1.0], dtype=np.float32),
            }

    class Extractor:
        def extract_residue_embeddings(self, sequence):
            return torch.ones((len(sequence), 4), dtype=torch.float32)

    class Reducer:
        def transform(self, embeddings):
            return embeddings

    count = _process_files(
        PreprocessConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=tmp_path / "unused-esm.pth",
        ),
        [input_path],
        Parser(),
        Extractor(),
        Reducer(),
    )

    manifest = json.loads((output_dir / PREPROCESS_MANIFEST).read_text(encoding="utf-8"))
    assert count == 1
    assert manifest["complete"] is True
    assert manifest["input_hashes_complete"] is True
    assert manifest["produced_outputs"] == ["sample.pt"]


def test_preprocess_interruption_invalidates_old_complete_manifest_before_output_mutation(tmp_path):
    from protcross.data.preprocess import PREPROCESS_MANIFEST, PreprocessConfig, _process_files

    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    data_dir.mkdir()
    output_dir.mkdir()
    first = data_dir / "a.pdb"
    second = data_dir / "b.pdb"
    first.write_text("new-a", encoding="utf-8")
    second.write_text("new-b", encoding="utf-8")
    (output_dir / "a.pt").write_bytes(b"old-a")
    (output_dir / "b.pt").write_bytes(b"old-b")
    (output_dir / PREPROCESS_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "protcross-preprocess-v1",
                "complete": True,
                "append_mode": False,
                "produced_outputs": ["a.pt", "b.pt"],
            }
        ),
        encoding="utf-8",
    )

    class InterruptingParser:
        def parse_file_with_labels(self, path):
            if path == second:
                raise KeyboardInterrupt("simulated process termination")
            return {
                "coords": np.zeros((1, 3), dtype=np.float32),
                "sequence": "A",
                "plddts": np.array([90.0], dtype=np.float32),
                "residue_ids": ["A_1"],
                "labels": np.array([1.0], dtype=np.float32),
            }

    class Extractor:
        def extract_residue_embeddings(self, sequence):
            return torch.ones((len(sequence), 4), dtype=torch.float32)

    class Reducer:
        def transform(self, embeddings):
            return embeddings

    config = PreprocessConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=tmp_path / "unused-esm.pth",
    )
    with pytest.raises(KeyboardInterrupt, match="simulated process termination"):
        _process_files(config, [first, second], InterruptingParser(), Extractor(), Reducer())

    manifest = json.loads((output_dir / PREPROCESS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["input_hashes_complete"] is False
    assert manifest["produced_outputs"] == []
    assert torch.load(output_dir / "a.pt", weights_only=False)["sequence"] == "A"
    assert (output_dir / "b.pt").read_bytes() == b"old-b"


def test_dataset_rejects_incomplete_preprocess_manifest(tmp_path):
    from protcross.data.dataset import PREPROCESS_MANIFEST

    (tmp_path / PREPROCESS_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "protcross-preprocess-v1",
                "complete": False,
                "append_mode": True,
                "produced_outputs": ["sample.pt"],
            }
        ),
        encoding="utf-8",
    )
    dataset = object.__new__(EvoPointDataset)
    dataset.root = str(tmp_path)

    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        dataset._validate_preprocess_manifest()


def test_dataset_rejects_incomplete_preprocess_before_pyg_cache_build(tmp_path, monkeypatch):
    from protcross.data.dataset import PREPROCESS_MANIFEST

    torch.save(
        {
            "x": torch.randn(2, 4),
            "pos": torch.randn(2, 3),
            "plddt": torch.ones(2, 1),
            "y": torch.ones(2),
        },
        tmp_path / "partial.pt",
    )
    (tmp_path / PREPROCESS_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "protcross-preprocess-v1",
                "complete": False,
                "append_mode": False,
                "produced_outputs": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        EvoPointDataset,
        "process",
        lambda self: (_ for _ in ()).throw(AssertionError("cache build must not run")),
    )

    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        EvoPointDataset(root=tmp_path, split="all")
    assert not (tmp_path / "processed").exists()


def test_dataset_can_load_unlabeled_target_domain(tmp_path):
    torch.save(
        {
            "x": torch.randn(4, 128),
            "pos": torch.randn(4, 3),
            "plddt": torch.ones(4, 1),
            "y": torch.zeros(4),
        },
        tmp_path / "target.pt",
    )

    dataset = EvoPointDataset(
        root=tmp_path,
        split="train",
        require_labels=False,
        require_positive_labels=False,
    )

    assert len(dataset) == 1
    assert dataset[0].x.shape == (4, 128)


def test_dataset_cache_files_isolate_label_filter_configurations(tmp_path):
    torch.save(
        {
            "x": torch.randn(4, 8),
            "pos": torch.randn(4, 3),
            "plddt": torch.ones(4, 1),
        },
        tmp_path / "unlabeled.pt",
    )

    unlabeled = EvoPointDataset(
        root=tmp_path,
        split="all",
        require_labels=False,
        require_positive_labels=False,
    )
    assert len(unlabeled) == 1
    assert Path(unlabeled.processed_paths[0]).name == "data_cache_all_all.pt"

    with pytest.raises(RuntimeError, match="No valid data"):
        EvoPointDataset(
            root=tmp_path,
            split="all",
            require_labels=True,
            require_positive_labels=False,
        )
    assert (tmp_path / "processed" / "data_cache_all_all.pt").exists()
    assert not (tmp_path / "processed" / "data_cache_all_labeled.pt").exists()


def test_dataset_cache_rejects_old_schema_and_manifest_config_mismatch(tmp_path):
    from protcross.data.dataset import DATASET_CACHE_SCHEMA

    torch.save(
        {
            "x": torch.randn(4, 8),
            "pos": torch.randn(4, 3),
            "plddt": torch.ones(4, 1),
            "y": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        },
        tmp_path / "sample.pt",
    )
    dataset = EvoPointDataset(root=tmp_path, split="all")
    manifest_path = Path(dataset.processed_paths[1])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == DATASET_CACHE_SCHEMA

    manifest["schema_version"] = "protcross-dataset-cache-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert dataset._cache_is_stale() is True

    manifest["schema_version"] = DATASET_CACHE_SCHEMA
    manifest["require_positive_labels"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert dataset._cache_is_stale() is True


def test_dataset_cache_rebuilds_when_source_pt_changes(tmp_path):
    def write_sample(value: float) -> None:
        torch.save(
            {
                "x": torch.full((4, 128), value),
                "pos": torch.randn(4, 3),
                "plddt": torch.ones(4, 1),
                "y": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            tmp_path / "sample.pt",
        )

    write_sample(1.0)
    dataset = EvoPointDataset(root=tmp_path, split="all")
    assert float(dataset[0].x[0, 0]) == 1.0

    time.sleep(0.001)
    write_sample(9.0)
    dataset = EvoPointDataset(root=tmp_path, split="all")

    assert float(dataset[0].x[0, 0]) == 9.0


def test_small_dataset_splits_are_disjoint_and_cover_all_files(tmp_path):
    for index in range(9):
        (tmp_path / f"sample-{index}.pt").write_bytes(b"sample")
    dataset = object.__new__(EvoPointDataset)
    dataset.root = str(tmp_path)
    dataset.split_seed = 42

    splits = {}
    for split in ("train", "val", "test"):
        dataset.split = split
        splits[split] = set(dataset._split_files())

    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])
    assert set.union(*splits.values()) == {str(path) for path in tmp_path.glob("*.pt")}


def test_dataset_cache_signature_hashes_each_source_once(tmp_path, monkeypatch):
    paths = [tmp_path / f"sample-{index}.pt" for index in range(3)]
    for path in paths:
        path.write_bytes(path.name.encode())
    dataset = object.__new__(EvoPointDataset)
    dataset.root = str(tmp_path)
    calls = []
    original = EvoPointDataset._file_signature

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(dataset, "_file_signature", counted)
    dataset._cache_signature([str(paths[0]), str(paths[1])])

    assert sorted(calls) == sorted(str(path) for path in paths)


def test_dataset_rejects_nonfinite_or_misaligned_payload(tmp_path):
    source = tmp_path / "invalid.pt"
    torch.save(
        {
            "x": torch.tensor([[float("nan"), 0.0], [1.0, 2.0]]),
            "pos": torch.zeros((2, 3)),
            "plddt": torch.ones((2, 1)),
            "y": torch.ones(2),
        },
        source,
    )
    dataset = object.__new__(EvoPointDataset)
    dataset.require_labels = True
    dataset.require_positive_labels = True

    assert dataset._load_data_file(str(source)) is None

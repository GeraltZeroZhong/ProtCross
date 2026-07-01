import pytest
import shlex
import sys

from protcross.experiments import strategy_search
from protcross.experiments.multiseed_benchmark import run_command


class _FakeModel:
    hparams = type("HParams", (), {"use_esm": True})()

    def eval(self):
        return self

    def to(self, device):
        return self


def test_strategy_search_evaluate_iou_refuses_implicit_train_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        strategy_search.EvoPointDALitModule,
        "load_from_checkpoint",
        staticmethod(lambda *args, **kwargs: _FakeModel()),
    )

    def fake_dataset(*, root, split):
        if split == "test":
            raise ValueError("missing test split")
        return []

    monkeypatch.setattr(strategy_search, "EvoPointDataset", fake_dataset)

    with pytest.raises(RuntimeError, match="refusing to evaluate on train split implicitly"):
        strategy_search.evaluate_iou("fake.ckpt", str(tmp_path), device="cpu")


def test_strategy_search_evaluate_iou_fails_on_model_load_error(monkeypatch, tmp_path):
    def fail_load(*args, **kwargs):
        raise ValueError("bad checkpoint")

    monkeypatch.setattr(strategy_search.EvoPointDALitModule, "load_from_checkpoint", staticmethod(fail_load))

    with pytest.raises(RuntimeError, match="Model load failed"):
        strategy_search.evaluate_iou("fake.ckpt", str(tmp_path), device="cpu")


def test_multiseed_run_command_uses_non_shell_command(tmp_path):
    log_file = tmp_path / "logs" / "command.log"

    output = run_command(f"{shlex.quote(sys.executable)} -c \"print('ok')\"", str(log_file))

    assert output.strip() == "ok"
    assert log_file.read_text(encoding="utf-8").strip() == "ok"


def test_strategy_checkpoint_resume_requires_env_and_matching_manifest(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    signature = {"strategy": "Standard_DANN", "seed": 42}
    strategy_search.write_checkpoint_manifest(str(checkpoint), signature)

    monkeypatch.delenv("PROTCROSS_STRATEGY_RESUME", raising=False)
    assert strategy_search.should_resume_checkpoint(str(checkpoint), signature) is False

    monkeypatch.setenv("PROTCROSS_STRATEGY_RESUME", "1")
    assert strategy_search.should_resume_checkpoint(str(checkpoint), signature) is True
    assert strategy_search.should_resume_checkpoint(str(checkpoint), {"strategy": "other"}) is False
    checkpoint.write_bytes(b"changed checkpoint")
    assert strategy_search.should_resume_checkpoint(str(checkpoint), signature) is False

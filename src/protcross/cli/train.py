"""Hydra CLI entry point for standard ProtCross training."""

from __future__ import annotations

import sys
from importlib import resources

CONFIG_PACKAGE = "protcross.configs"


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Run Hydra training from a console script or the unified dispatcher."""
    command_args = list(sys.argv[1:] if argv is None else argv)
    command_prog = prog or "protcross-train"
    if any(arg in {"-h", "--help"} for arg in command_args):
        print(_format_help(command_prog))
        return 0

    import hydra
    from omegaconf import DictConfig

    config_path = str(resources.files(CONFIG_PACKAGE))

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train")
    def _hydra_main(cfg: DictConfig) -> None:
        from protcross.training import run_training

        run_training(cfg)

    old_argv = sys.argv[:]
    if argv is not None:
        sys.argv = [command_prog, *argv]
    try:
        _hydra_main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        return code
    finally:
        sys.argv = old_argv
    return 0


def _format_help(prog: str) -> str:
    return f"""usage: {prog} [HYDRA_OVERRIDES...]

Run the ProtCross Hydra training workflow.

Common overrides:
  trainer.max_epochs=10
  trainer.accelerator=cpu
  data.data_dir_pdb=data/processed_pdb
  data.data_dir_af2=data/processed_af2
  model.use_da=False

Hydra syntax is forwarded unchanged at execution time. Install the training
dependencies and run `{prog} --hydra-help` for Hydra's full help output.
"""


if __name__ == "__main__":
    raise SystemExit(main())

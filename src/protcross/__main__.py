"""Allow ``python -m protcross`` when console-script PATH setup is unavailable."""

from protcross.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())

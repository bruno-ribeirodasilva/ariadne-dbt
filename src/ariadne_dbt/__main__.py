"""Entry point for `ariadne` CLI and `python -m ariadne_dbt`."""

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

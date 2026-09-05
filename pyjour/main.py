"""Command-line interface for the standalone Python package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from . import __version__, infer_detailed


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argument_parser()
    parsed = parser.parse_args(arguments)
    display_name = " ".join(parsed.display_name)
    try:
        detailed = infer_detailed(
            display_name,
            country_hint=parsed.country,
            locale_hint=parsed.locale,
            gender_hint=parsed.gender,
        )
    except ValueError as error:
        parser.error(str(error))

    if parsed.json:
        print(json.dumps(detailed, ensure_ascii=False, indent=2))
        return
    print(f"Bonjour {detailed['greeting_name'] or display_name} !")


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyjour",
        description="Infer a conservative greeting name from a display name.",
    )
    parser.add_argument("display_name", nargs="+", metavar="DISPLAY_NAME")
    parser.add_argument("--country", metavar="XX", help="two-letter country hint")
    parser.add_argument("--locale", help="locale used as country fallback")
    parser.add_argument("--gender", metavar="F|M", help="gender hint")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print detailed inference as JSON",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


if __name__ == "__main__":
    main()

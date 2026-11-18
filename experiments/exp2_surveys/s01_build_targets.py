"""Build processed GSS, WVS, or ANES targets for Experiment 2."""

from __future__ import annotations

import argparse

from mode_collapse.surveys.targets import build_anes, build_gss, build_wvs

BUILDERS = {
    "gss": build_gss,
    "wvs": build_wvs,
    "anes": build_anes,
}


def main() -> None:
    """Reads one raw survey file and writes its processed targets."""
    # --- config ---
    parser = argparse.ArgumentParser(description="Build Experiment 2 survey targets.")
    parser.add_argument("--dataset", required=True, choices=["gss", "wvs", "anes"])
    args = parser.parse_args()

    # --- data import ---
    BUILDERS[args.dataset]()


if __name__ == "__main__":
    main()

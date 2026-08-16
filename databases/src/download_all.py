#!/usr/bin/env python3
"""Recreate the entire local antibody-design database from scratch on any machine.

Runs every source's fetch script in order (cheap/free ones first). This is the single
entry point for "I'm on a new computer, get me back to where this project's databases/
was" — see databases/README.md for what each source contains.

Usage:
    python3 databases/src/download_all.py                  # everything, ANDD structures skipped
    python3 databases/src/download_all.py --andd-structures  # also ANDD's 2.2GB structures zip
    python3 databases/src/download_all.py --skip abdesign_db --skip asd   # e.g. skip the two
                                                              # non-commercial-license sources
"""

import argparse
import sys

import aacdb
import ab_bind
import abdesign_db
import andd
import asd
import sabdab

SOURCES = ["sabdab", "aacdb", "ab_bind", "andd", "abdesign_db", "asd"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip", action="append", default=[], choices=SOURCES, help="Skip a source (repeatable)")
    parser.add_argument("--andd-structures", action="store_true", help="Also fetch ANDD's 2.2GB structures zip (98%% overlaps sabdab/, skipped by default)")
    args = parser.parse_args()
    skip = set(args.skip)

    if "sabdab" not in skip:
        print("\n=== sabdab ===", file=sys.stderr)
        sabdab.fetch_summary()
        sabdab.fetch_affinity()
        sabdab.fetch_structures()

    if "aacdb" not in skip:
        print("\n=== aacdb ===", file=sys.stderr)
        aacdb.fetch_all()

    if "ab_bind" not in skip:
        print("\n=== ab_bind ===", file=sys.stderr)
        ab_bind.fetch()

    if "andd" not in skip:
        print("\n=== andd ===", file=sys.stderr)
        andd.fetch_metadata()
        if args.andd_structures:
            andd.fetch_structures()

    if "abdesign_db" not in skip:
        print("\n=== abdesign_db (CC BY-NC 4.0) ===", file=sys.stderr)
        abdesign_db.fetch()

    if "asd" not in skip:
        print("\n=== asd (non-commercial research use) ===", file=sys.stderr)
        asd.fetch()


if __name__ == "__main__":
    main()

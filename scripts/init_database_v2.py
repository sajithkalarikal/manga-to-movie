from __future__ import annotations

import argparse
from pathlib import Path

from modules.database_v2 import initialize_v2_database, list_v2_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the next-step SQLite schema without wiring it into the app.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("outputs/app_state_v2.sqlite3"),
        help="Path to the v2 SQLite database file.",
    )
    args = parser.parse_args()

    database_path = initialize_v2_database(args.database)
    tables = list_v2_tables(database_path)

    print(f"Initialized v2 database at: {database_path}")
    print("Tables:")
    for table in tables:
        print(f"- {table}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test MySQL connectivity using the same settings as final.py (.env: MYSQL_*).
Reports whether the user appears to have read-only or read-write access.

Usage: python check_db_connection.py
"""

import os
import re
import sys

from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error as MySQLError

load_dotenv()


def _infer_access_from_grants(grant_strings: list[str]) -> tuple[str, str]:
    """
    Infer read-only vs read-write from SHOW GRANTS output.
    Returns (label, detail) e.g. ("read-write", "Grants include INSERT/UPDATE/DELETE or ALL PRIVILEGES.")
    """
    text = " ".join(grant_strings).upper()
    # Global or schema-level full access
    if "ALL PRIVILEGES" in text:
        return "read-write", "Grants include ALL PRIVILEGES."

    # Data-modifying privileges (any schema or specific DB)
    write_keywords = ("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "REFERENCES", "TRIGGER")
    found_write = [kw for kw in write_keywords if re.search(rf"\b{kw}\b", text)]
    if found_write:
        return "read-write", f"Grants include: {', '.join(found_write)}."

    # Often read-only users get SELECT + USAGE only
    if "SELECT" in text and not found_write:
        return "read-only", "Grants appear to be SELECT (and possibly USAGE) only; no INSERT/UPDATE/DELETE detected."

    return "unknown", "Could not classify from grants; review SHOW GRANTS output below."


def _probe_temp_write(cursor) -> tuple[bool, str]:
    """
    Try CREATE TEMPORARY TABLE + INSERT + DROP. Does not touch permanent tables.
    Returns (success, message).
    """
    name = "_conn_write_probe_tmp"
    try:
        cursor.execute(
            f"CREATE TEMPORARY TABLE {name} (id TINYINT UNSIGNED NOT NULL)"
        )
        cursor.execute(f"INSERT INTO {name} (id) VALUES (1)")
        cursor.execute(f"DROP TEMPORARY TABLE {name}")
        return True, "Temporary table create/insert/drop succeeded (session has DDL + INSERT on temp objects)."
    except MySQLError as e:
        return False, f"Write probe failed: {e}"


def main() -> int:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")

    missing = [k for k, v in [
        ("MYSQL_USER", user),
        ("MYSQL_PASSWORD", password),
        ("MYSQL_DATABASE", database),
    ] if not v]
    if missing:
        print("Missing in .env:", ", ".join(missing))
        print("Set MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE.")
        return 1

    print(f"Connecting to {host}:{port} / database={database!r} as {user!r} ...")

    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="latin1",
        )
        cursor = conn.cursor()

        # --- Read access ---
        cursor.execute("SELECT 1 AS ok")
        row = cursor.fetchone()
        print("Read test (SELECT 1):", row, "-> OK")

        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        print("Current database:", db_name)

        # --- Privileges from server ---
        cursor.execute("SHOW GRANTS FOR CURRENT_USER()")
        grant_rows = [r[0] for r in cursor.fetchall()]
        print("\nSHOW GRANTS FOR CURRENT_USER():")
        for g in grant_rows:
            print(" ", g)

        label, detail = _infer_access_from_grants(grant_rows)
        print(f"\nInferred from grants: {label}")
        print(f"  ({detail})")

        # --- Practical write probe (temp table only) ---
        ok_probe, probe_msg = _probe_temp_write(cursor)
        print(f"\nWrite probe (temporary table): {'OK' if ok_probe else 'FAILED'}")
        print(f"  {probe_msg}")

        # Summary: grants say read-only but temp write works -> note mismatch
        if label == "read-only" and ok_probe:
            print(
                "\nNote: Grants look read-only for permanent tables, but temporary table "
                "create/insert worked. Check whether INSERT/UPDATE on real tables is allowed."
            )
        elif label == "read-write" and not ok_probe:
            print(
                "\nNote: Grants suggest write access, but the temp-table probe failed. "
                "The account may lack CREATE TEMPORARY or session limits may apply."
            )

        cursor.close()
        conn.close()
        print("\nConnection closed. Overall: database connection is working.")
        return 0
    except MySQLError as e:
        print("MySQL error:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

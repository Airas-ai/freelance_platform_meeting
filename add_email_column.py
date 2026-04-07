#!/usr/bin/env python3
"""
Add `email` column to `ai_appointment_agent` if it does not exist.

Uses MYSQL_* from .env (same as test_agent.py / final.py).

Usage: python add_email_column.py
"""

import os

from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error as MySQLError

load_dotenv()

ALTER_SQL = """
ALTER TABLE ai_appointment_agent
ADD COLUMN email VARCHAR(255) NULL AFTER last_name
"""


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
        return 1

    print(f"Connecting to {host}:{port} / database={database!r} ...")

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
        cursor.execute(ALTER_SQL.strip())
        conn.commit()
        cursor.close()
        conn.close()
        print("OK: Column `email` added to `ai_appointment_agent`.")
        return 0
    except MySQLError as e:
        err = str(e)
        if "1060" in err or "Duplicate column name" in err or "duplicate column" in err.lower():
            print("OK: Column `email` already exists; nothing to do.")
            return 0
        print("MySQL error:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

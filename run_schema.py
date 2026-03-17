#!/usr/bin/env python3
"""
Run schema.sql against your MySQL database using credentials from .env.
Usage: python run_schema.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import mysql.connector

load_dotenv()

def main():
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")

    missing = [k for k, v in [("MYSQL_USER", user), ("MYSQL_PASSWORD", password), ("MYSQL_DATABASE", database)] if not v]
    if missing:
        print("Missing in .env:", ", ".join(missing))
        print("Add MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DATABASE to your .env file.")
        return 1

    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()

    def strip_leading_comments(text: str) -> str:
        """Remove leading lines that are blank or only -- comments."""
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s and not s.startswith("--"):
                start = i
                break
        return "\n".join(lines[start:]).strip()

    # Split by semicolon; strip leading comments from each chunk so CREATE TABLE is not skipped
    statements = []
    for part in sql.split(";"):
        stmt = strip_leading_comments(part).strip()
        if stmt:
            statements.append(stmt)

    try:
        print(f"Connecting to: {host}:{port} / database = '{database}'")
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        cursor = conn.cursor()
        for stmt in statements:
            if stmt:
                cursor.execute(stmt)
                print("Executed:", stmt[:50].replace("\n", " ") + "...")
        conn.commit()

        # Verify: list tables in this database
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        print("Tables in this database:", tables)
        if "meeting_appointments" in tables:
            print("✓ meeting_appointments exists.")
        else:
            print("⚠ meeting_appointments not in list (check table name / database).")

        cursor.close()
        conn.close()
        print("Schema applied successfully.")
        print("In DBeaver: refresh the database (right-click → Refresh) and ensure you're viewing database:", repr(database))
        return 0
    except mysql.connector.Error as e:
        print("MySQL error:", e)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

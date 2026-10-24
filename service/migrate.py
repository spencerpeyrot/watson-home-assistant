"""Apply durable PostgreSQL migrations from DATABASE_URL process setting."""

from __future__ import annotations

import asyncio
import os

from service.postgres import PostgresStore


async def migrate(database_url: str) -> tuple[str, ...]:
    if not database_url.strip():
        raise ValueError("DATABASE_URL is required")
    store = PostgresStore(database_url)
    await store.open()
    try:
        return await store.apply_migrations()
    finally:
        await store.close()


def main() -> int:
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url.strip():
        print("DATABASE_URL is required in process environment.")
        return 2
    try:
        applied = asyncio.run(migrate(database_url))
    except Exception as error:
        print(f"Migration failed ({type(error).__name__}); connection details suppressed.")
        return 1
    print("Applied migrations: " + (", ".join(applied) if applied else "none needed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

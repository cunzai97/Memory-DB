"""Memory-DB management CLI — backup, export, import, rebuild index.

Usage:
    python -m memory_simple.manage list [--project-id X] [--limit 100]
    python -m memory_simple.manage export --path backups/memories.json
    python -m memory_simple.manage import --path backups/memories.json [--dedup-threshold 0.85]
    python -m memory_simple.manage rebuild --embedding-url http://localhost:9090/v1/embeddings
    python -m memory_simple.manage purge --min-recall-count 0 --unused-days 30
    python -m memory_simple.manage delete-all [--project-id X]
"""

import argparse
import asyncio
import json

from memory_simple.admin import MemoryAdmin


async def cmd_list(args):
    admin = MemoryAdmin()
    memories = await admin.list_memories(
        limit=args.limit,
        offset=0,
    )
    for m in memories:
        print(json.dumps(m, ensure_ascii=False))


async def cmd_export(args):
    admin = MemoryAdmin()
    result = await admin.export_to_json(args.path)
    print(json.dumps(result, ensure_ascii=False))


async def cmd_import(args):
    admin = MemoryAdmin()
    result = await admin.import_from_json(
        path=args.path,
        project_id=args.project_id or None,
        dedup_threshold=args.dedup_threshold,
    )
    print(json.dumps(result, ensure_ascii=False))


async def cmd_rebuild(args):
    admin = MemoryAdmin()
    result = await admin.rebuild_index(
        new_embedding_url=args.embedding_url or None,
    )
    print(json.dumps(result, ensure_ascii=False))


async def cmd_delete_all(args):
    if not args.force:
        confirm = input("This will delete all memories. Type 'DELETE' to confirm: ")
        if confirm != "DELETE":
            print("Cancelled.")
            return

    admin = MemoryAdmin()
    result = await admin.delete_all()
    print(json.dumps(result, ensure_ascii=False))


async def cmd_purge(args):
    admin = MemoryAdmin()

    # Show preview in dry-run mode (default)
    if not args.execute:
        result = await admin.purge_unused(
            min_recall_count=args.min_recall_count,
            unused_days=args.unused_days,
            dry_run=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("would_delete", 0) > 0:
            print("\nRun with --execute to actually delete.")
    else:
        # Require confirmation for actual deletion
        if not args.force:
            result = await admin.purge_unused(
                min_recall_count=args.min_recall_count,
                unused_days=args.unused_days,
                dry_run=True,
            )
            count = result.get("would_delete", 0)
            criteria = result.get("criteria", {})
            print(f"Will delete {count} memories matching: {criteria}")
            confirm = input("Type 'PURGE' to confirm: ")
            if confirm != "PURGE":
                print("Cancelled.")
                return

        result = await admin.purge_unused(
            min_recall_count=args.min_recall_count,
            unused_days=args.unused_days,
            dry_run=False,
        )
        print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(prog="memory-db-manage", description="Memory-DB management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List all memories (no search)")
    p_list.add_argument("--limit", type=int, default=100, help="Max results per page")

    # export
    p_export = subparsers.add_parser("export", help="Export all memories to JSON file")
    p_export.add_argument("--path", required=True, help="Output file path (e.g. backups/memories.json)")

    # import
    p_import = subparsers.add_parser("import", help="Import memories from JSON export file")
    p_import.add_argument("--path", required=True, help="Input file path")
    p_import.add_argument("--dedup-threshold", type=float, default=0.85, help="Dedup threshold (0 to disable)")

    # rebuild
    p_rebuild = subparsers.add_parser("rebuild", help="Re-encode all memories with current/new embedding model")
    p_rebuild.add_argument("--embedding-url", default=None, help="New embedding API URL (keeps current if not set)")

    # delete-all
    p_delete = subparsers.add_parser("delete-all", help="Delete all memories (destructive!)")
    p_delete.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # purge
    p_purge = subparsers.add_parser("purge", help="Purge unused memories based on recall stats")
    p_purge.add_argument("--min-recall-count", type=int, default=None, help="Delete memories with recall_count <= N (default: no filter)")
    p_purge.add_argument("--unused-days", type=int, default=None, help="Delete memories not recalled in N days (default: no filter)")
    p_purge.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")
    p_purge.add_argument("--force", action="store_true", help="Skip confirmation prompt (with --execute)")

    args = parser.parse_args()
    handlers = {
        "list": cmd_list,
        "export": cmd_export,
        "import": cmd_import,
        "rebuild": cmd_rebuild,
        "delete-all": cmd_delete_all,
        "purge": cmd_purge,
    }
    asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    main()

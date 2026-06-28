"""Memory-DB — minimal AI memory system.

Core:
  - MemoryService: store, get, update (for MCP tools)
  - MemoryAdmin: list, export, import, rebuild, purge (for CLI)
"""

from memory_simple.service import MemoryService
from memory_simple.admin import MemoryAdmin

__all__ = ["MemoryService", "MemoryAdmin"]

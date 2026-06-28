# Memory-DB — 极简 AI 记忆系统

一个轻量级的向量记忆系统，专为 AI Agent 设计。存储、检索、删除记忆，基于语义相似度搜索。你拥有嵌入模型，我们处理向量。

## 为什么选择 Memory-DB？

**最小 token 消耗，最大模型自由度。**

- **~250 tokens/轮** — 4 个工具，描述精简，不写冗长指令
- **无知识图谱、无全文搜索、无仪表盘** — 模型决定如何使用记忆，而非系统限定
- **6 个文件，3 个依赖** — 轻量 footprint，易懂易改

**你自己提供嵌入 API。**

- 使用 llama.cpp、Ollama、OpenAI 或任何 OpenAI 兼容端点
- 随时切换模型 —— 只需更改 `EMBEDDING_API_URL`
- 切换模型后用 `memory-db-manage rebuild` 重建向量

我们移除了一切束缚模型的冗余。记忆系统应该是工具，而非框架。

## 架构

```
┌──────────────┐      ┌─────────────┐       ┌──────────────┐
│  MCP Server   │─────▶│    Qdrant   │◀──────│              │
│  (3 tools)    │◀─────│  :6333      │       │ llama.cpp    │
└──────────────┘      └─────────────┘       │  :8081       │
                                          │  /v1/embed   │
                                          └──────────────┘
```

一个向量数据库，一个嵌入 API。4 个 MCP 工具给 AI Agent，一个管理 CLI 给运维。

## 安装

```bash
# 克隆仓库
git clone https://github.com/cunzai97/Memory-DB.git
cd Memory-DB

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows 上用: venv\Scripts\activate

# 安装依赖
pip install -e .

# 启动 Qdrant（必需）
docker compose up -d
```

现在可以启动 MCP server：

```bash
memory-db  # 启动 MCP server
```

或使用管理 CLI：

```bash
memory-db-manage list  # 列出所有记忆
```

## 快速开始

### 前置条件

- Qdrant 运行在 `:6333`（Docker Compose 或独立部署）
- 嵌入 API 运行在 `:8081`（你的 llama.cpp 实例）

```bash
docker compose up -d          # 仅启动 Qdrant
pip install -e .              # 安装 memory-db + CLI
memory-db                     # 启动 MCP server
```

### 环境变量

| 变量 | 默认值 | 说明 |
|----------|---------|---------|
| `QDRANT_HOST` | `localhost` | Qdrant 主机 |
| `QDRANT_PORT` | `6333` | Qdrant 端口 |
| `EMBEDDING_API_URL` | `http://localhost:8081/v1/embeddings` | 嵌入 API（OpenAI 兼容） |

## MCP 配置

### Claude Code — 全局配置

```bash
claude mcp add memory-db \
  -e PYTHONPATH=/path/to/Memory-DB/src \
  -e EMBEDDING_API_URL=http://localhost:8081/v1/embeddings \
  -e QDRANT_HOST=localhost \
  -e QDRANT_PORT=6333 \
  -- /path/to/Memory-DB/venv/bin/python3 -m memory_simple.server
```

验证：`claude mcp list`（应显示 `✓ Connected`）。

### Hermes — 全局配置

编辑 `~/.hermes/config.yaml`，在 `mcp_servers` 下添加：

```yaml
mcp_servers:
  memory-db:
    command: /path/to/Memory-DB/venv/bin/python3
    args: ["-m", "memory_simple.server"]
    timeout: 120
    env:
      PYTHONPATH: /path/to/Memory-DB/src
      EMBEDDING_API_URL: http://localhost:8081/v1/embeddings
      QDRANT_HOST: localhost
      QDRANT_PORT: "6333"
```

重启 Hermes。

## MCP 工具

### `store_memory(content, tags?, dedup_threshold=0.85)`

保存记忆。返回 `{id, deduped}`。

如果文本与已有记忆语义相似（余弦相似度 ≥ 阈值），旧记忆会被替换——内置熵减机制。设置 `dedup_threshold=0` 可禁用去重。

```
store_memory(content="Python是动态类型语言")
→ {"id": "a1b2c3d4-...", "deduped": false}

store_memory(content="Python是一门动态类型的编程语言")
→ {"id": "e5f6g7h8-...", "deduped": true}  // 替换了重复记忆
```

### `get_memories(query, limit=5, min_score=0.5)`

语义搜索（余弦相似度）。返回按相似度降序排列的结果。低于 `min_score` 的结果会被过滤——用 0.8+ 做严格匹配，用低值（如 0.2）做广泛搜索。每次命中自动递增 `recall_count` 并更新 `last_recalled_at`。

```
get_memories(query="动态类型")
→ [{"id": "...", "content": "Python是动态类型语言", "score": 0.79,
     "recall_count": 1, "last_recalled_at": "2026-06-27T...", ...}]

# 更广泛的搜索，降低阈值
get_memories(query="动态类型", min_score=0.2)
```

### `update_memory(memory_id, content?, oldText?, newText?, tags?)`

按 ID 更新记忆的内容和/或标签。支持两种模式：
- **完整替换**: `content="..."` — 替换整个内容，向量重新编码。
- **部分替换**: `oldText="匹配" newText="替换"` — 查找精确子串并替换，向量重新编码。

content、tags 或 (oldText+newText) 至少提供一个。`content` 和 `(oldText+newText)` 互斥。返回 `{updated: true, id, changes, update_type}`。

```
update_memory(memory_id="a1b2c3d4-...", content="更新后的文本")
→ {"updated": true, "id": "a1b2c3d4-...", "changes": {"content": true}, "update_type": "full_replace"}

update_memory(memory_id="a1b2c3d4-...", oldText="旧文本", newText="新文本")
→ {"updated": true, "id": "a1b2c3d4-...", "changes": {"content": true}, "update_type": "partial_replace"}

update_memory(memory_id="a1b2c3d4-...", tags=["新标签"])
→ {"updated": true, "id": "a1b2c3d4-...", "changes": {"tags": true}, "update_type": "tags_only"}
```

### `delete_memory(memory_id)`

按 ID 删除记忆。返回 `{deleted: true, id}` 或 `{deleted: false, id, error}`。

```
delete_memory(memory_id="a1b2c3d4-...")
→ {"deleted": true, "id": "a1b2c3d4-..."}
```

### Embedding API 限制

内容必须 <1024 tokens（embedding API 限制），超出会报 400 错误。

### Token 成本

MCP 工具定义（描述 + JSON schemas）每轮消耗约 **~250 tokens** 的系统提示词——四个工具共约 600 字符的描述文本。自解释参数（`content`、`tags`、`query`、`limit`）不写说明，只有需要解释的参数（`dedup_threshold`、`min_score`）才有描述。这是每次请求的一次性开销，不会累积。

## 管理 CLI

通过终端进行管理操作——不暴露为 MCP 工具。

```bash
# 列出所有记忆（无搜索）
memory-db-manage list [--limit N]

# 导出为 JSON 备份（保留原始文本，与向量无关）
memory-db-manage export --path backups/memories.json

# 从 JSON 导入（使用当前嵌入模型重新编码）
memory-db-manage import --path backups/memories.json

# 重建索引——使用相同或新模型重新编码所有记忆
memory-db-manage rebuild [--embedding-url http://new-host:port/v1/embeddings]

# 清理未使用的记忆（熵减）
memory-db-manage purge --min-recall-count 0 --unused-days 30   # 默认 dry-run
memory-db-manage purge --min-recall-count 0 --unused-days 30 --execute  # 实际删除

# 删除全部（危险操作，需确认）
memory-db-manage delete-all [--force]
```

### 切换嵌入模型

更换嵌入模型后，已有向量会失效。两种方案：

1. **原地重建** —— 保留元数据和召回统计，仅替换向量：
   ```bash
   memory-db-manage rebuild --embedding-url http://new-host:9090/v1/embeddings
   ```

2. **导出 → 导入** —— 完整文本备份：
   ```bash
   memory-db-manage export --path backups/old-model.json
   # ... 切换模型 ...
   memory-db-manage import --path backups/old-model.json
   ```

### 熵减（清理未使用记忆）

从未被召回的记忆会随时间累积。使用 `purge` 清理：

```bash
# 预览将被删除的内容（默认 dry-run）
memory-db-manage purge --min-recall-count 0 --unused-days 30

# 实际删除 recall_count=0 且超过 30 天未被召回的记忆
memory-db-manage purge --min-recall-count 0 --unused-days 30 --execute
```

## 记忆 Payload 结构

每条存储的记忆在 Qdrant 中携带以下 payload：

```json
{
  "id": "<uuid>",
  "content": "原始文本",
  "created_at": "2026-06-27T14:33:24+00:00",
  "tags": ["rust", "systems"],
  "recall_count": 3,
  "last_recalled_at": "2026-06-27T15:00:00+00:00"
}
```

`tags` 可选。`recall_count` / `last_recalled_at` 在每次搜索命中时自动更新——用于识别熵减时从未被召回的记忆。

## 项目结构

```
src/memory_simple/
├── embedding.py   # 嵌入 API 客户端 (httpx)
├── service.py     # MemoryService — 核心 store/get/update
├── admin.py       # MemoryAdmin — backup/import/rebuild/purge
├── server.py      # MCP server — 暴露 3 个工具
└── manage.py      # CLI — 管理操作
```

**依赖：** `mcp`, `qdrant-client`, `httpx` —— 仅此而已。

# ZotSeek Python Client

Minimal Python client for ZotSeek's MCP server.

## Installation

```bash
pip install httpx
```

## Usage

```python
import asyncio
from zotseek_mcp_client import ZotSeekClient, SearchMode, Granularity

async def main():
    async with ZotSeekClient() as client:
        # Check index
        status = await client.get_index_status()
        print(f"Ready: {status.ready}, Papers: {status.indexed_papers}")
        
        # Search
        results = await client.search("machine learning", max_results=5)
        for r in results:
            print(f"{r.title}: {r.score:.4f}")
            if r.matched_chunk.snippet:
                print(f"  {r.matched_chunk.snippet[:100]}...")

asyncio.run(main())
```

## Requirements

- Zotero running with ZotSeek plugin
- AI Agent Access enabled in ZotSeek settings
- HTTP server enabled in Zotero settings
- Library indexed

## Testing

See `tests/test_retrieval_performance.py` for simple retrieval tests.
Customize `EXAMPLE_TESTS` with your actual papers and run:

```bash
python tests/test_retrieval_performance.py
```

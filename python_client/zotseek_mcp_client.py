#!/usr/bin/env python3
"""
Minimal ZotSeek MCP Client for Python

Simple client for interacting with ZotSeek's MCP server to perform
semantic search and retrieval on scientific publications.

Usage:
    from zotseek_mcp_client import ZotSeekClient
    async with ZotSeekClient() as client:
        results = await client.search("machine learning")
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx


class SearchMode(str, Enum):
    """Search mode for ZotSeek queries."""
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class Granularity(str, Enum):
    """Result granularity for search."""
    PAPERS = "papers"
    PASSAGES = "passages"


@dataclass
class MatchedChunk:
    """Matched text chunk with location info."""
    snippet: Optional[str] = None
    page: Optional[int] = None
    text_source: Optional[str] = None


@dataclass
class Links:
    """Deep links to open results in Zotero."""
    select: Optional[str] = None
    select_http: Optional[str] = None
    open_pdf: Optional[str] = None
    open_pdf_http: Optional[str] = None


@dataclass
class Result:
    """A single search result."""
    item_key: str
    library_key: Optional[str] = None
    title: str = ""
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    score: float = 0.0
    source: Optional[str] = None
    matched_chunk: MatchedChunk = field(default_factory=MatchedChunk)
    links: Links = field(default_factory=Links)


@dataclass 
class IndexStatus:
    """ZotSeek index status."""
    ready: bool = False
    model_loaded: bool = False
    indexed_papers: int = 0
    total_chunks: int = 0
    model_id: str = ""
    storage_used_bytes: int = 0


class ZotSeekClient:
    """
    Minimal client for ZotSeek MCP server.
    
    Connects to ZotSeek's local HTTP server to perform search operations.
    """
    
    DEFAULT_URL = "http://localhost:23119"
    MCP_PATH = "/zotseek/mcp"
    
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, *args):
        await self.close()
        
    async def connect(self):
        """Connect to ZotSeek MCP server."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        # Initialize MCP session
        await self._rpc("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "zotseek-client", "version": "1.0"}
        })
        
    async def close(self):
        """Close connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            
    async def _rpc(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request."""
        if not self._client:
            raise RuntimeError("Not connected")
            
        response = await self._client.post(
            urljoin(self.base_url, self.MCP_PATH),
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            
        result = response.json()
        if "error" in result:
            raise RuntimeError(f"RPC error: {result['error'].get('message', '')}")
            
        return result.get("result", {})
        
    async def _tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        if "content" in result and result["content"]:
            content = result["content"][0]
            if content.get("type") == "text":
                return json.loads(content["text"])
        return result
        
    async def get_index_status(self) -> IndexStatus:
        """Get index status."""
        data = await self._tool("index_status", {})
        return IndexStatus(
            ready=data.get("ready", False),
            model_loaded=data.get("modelLoaded", False),
            indexed_papers=data.get("indexedPapers", 0),
            total_chunks=data.get("totalChunks", 0),
            model_id=data.get("modelId", ""),
            storage_used_bytes=data.get("storageUsedBytes", 0)
        )
        
    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        max_results: int = 10,
        granularity: Granularity = Granularity.PAPERS
    ) -> List[Result]:
        """Search the library."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        data = await self._tool("search", {
            "query": query.strip(),
            "max_results": max_results,
            "mode": mode.value,
            "granularity": granularity.value
        })
        
        results = []
        for item in data.get("results", []):
            # Parse authors (can be string or list)
            authors = item.get("authors")
            if isinstance(authors, str):
                authors = [authors] if authors else None
            
            # Parse matched chunk
            chunk_data = item.get("matchedChunk", {}) or {}
            matched_chunk = MatchedChunk(
                snippet=chunk_data.get("snippet"),
                page=chunk_data.get("page"),
                text_source=chunk_data.get("textSource")
            )
            
            # Parse links
            links_data = item.get("links", {}) or {}
            links = Links(
                select=links_data.get("select"),
                select_http=links_data.get("selectHttp"),
                open_pdf=links_data.get("openPdf"),
                open_pdf_http=links_data.get("openPdfHttp")
            )
            
            results.append(Result(
                item_key=item.get("itemKey", ""),
                library_key=item.get("libraryKey"),
                title=item.get("title", ""),
                authors=authors,
                year=item.get("year"),
                score=item.get("score", 0.0),
                source=item.get("source"),
                matched_chunk=matched_chunk,
                links=links
            ))
            
        return results
        
    async def find_similar(
        self,
        item_key: str,
        library_key: str = "user",
        max_results: int = 10
    ) -> List[Result]:
        """Find papers similar to a known item."""
        if not item_key or len(item_key) != 8:
            raise ValueError("item_key must be 8 characters")
            
        data = await self._tool("find_similar", {
            "item_key": item_key.upper(),
            "library_key": library_key,
            "max_results": max_results
        })
        
        results = []
        for item in data.get("results", []):
            authors = item.get("authors")
            if isinstance(authors, str):
                authors = [authors] if authors else None
                
            chunk_data = item.get("matchedChunk", {}) or {}
            matched_chunk = MatchedChunk(
                snippet=chunk_data.get("snippet"),
                page=chunk_data.get("page"),
                text_source=chunk_data.get("textSource")
            )
            
            links_data = item.get("links", {}) or {}
            links = Links(
                select=links_data.get("select"),
                select_http=links_data.get("selectHttp"),
                open_pdf=links_data.get("openPdf"),
                open_pdf_http=links_data.get("openPdfHttp")
            )
            
            results.append(Result(
                item_key=item.get("itemKey", ""),
                library_key=item.get("libraryKey"),
                title=item.get("title", ""),
                authors=authors,
                year=item.get("year"),
                score=item.get("score", 0.0),
                source=None,  # find_similar doesn't have source
                matched_chunk=matched_chunk,
                links=links
            ))
            
        return results


# Simple convenience function
async def quick_search(query: str, base_url: str = ZotSeekClient.DEFAULT_URL) -> List[Result]:
    """Quick search without managing client."""
    async with ZotSeekClient(base_url=base_url) as client:
        return await client.search(query)

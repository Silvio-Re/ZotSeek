#!/usr/bin/env python3
"""
Simple test suite for ZotSeek retrieval performance.

Tests verify that ZotSeek can correctly:
1. Retrieve the correct document given a known query
2. Retrieve the correct paragraph/page from a document
3. Find similar documents

Run manually: python tests/test_retrieval_performance.py
Customize EXAMPLE_TESTS with your actual papers.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zotseek_mcp_client import ZotSeekClient, SearchMode, Granularity, Result


class Test:
    """Simple test base class."""
    
    def __init__(self, name, query, check_func):
        self.name = name
        self.query = query
        self.check_func = check_func  # Function that takes List[Result] and returns bool
        self.passed = False
        self.error = None
        self.results = []
        
    async def run(self, client):
        """Run the test."""
        try:
            self.results = await client.search(
                query=self.query,
                max_results=5
            )
            self.passed = self.check_func(self.results)
            if not self.passed:
                self.error = "Check function returned False"
        except Exception as e:
            self.error = str(e)
            self.passed = False


class ParagraphTest(Test):
    """Test paragraph retrieval with passage granularity."""
    
    def __init__(self, name, query, expected_text, check_func=None, 
                 expected_page=None, expected_section=None):
        def default_check(results):
            for r in results:
                if r.matched_chunk and r.matched_chunk.snippet:
                    if expected_text.lower() in r.matched_chunk.snippet.lower():
                        if expected_page and r.matched_chunk.page != expected_page:
                            continue
                        if expected_section and r.matched_chunk.text_source != expected_section:
                            continue
                        return True
            return False
            
        check = check_func or default_check
        super().__init__(name, query, check)
        self.expected_text = expected_text
        self.expected_page = expected_page
        self.expected_section = expected_section
        
    async def run(self, client):
        """Run with passage granularity."""
        try:
            self.results = await client.search(
                query=self.query,
                max_results=10,
                granularity=Granularity.PASSAGES
            )
            self.passed = self.check_func(self.results)
            if not self.passed:
                self.error = "Check function returned False"
        except Exception as e:
            self.error = str(e)
            self.passed = False


class SimilarTest:
    """Test similar document finding."""
    
    def __init__(self, name, item_key, expected_title, min_score=0.5):
        self.name = name
        self.item_key = item_key
        self.expected_title = expected_title.lower()
        self.min_score = min_score
        self.passed = False
        self.error = None
        self.results = []
        
    async def run(self, client):
        """Run the test."""
        try:
            self.results = await client.find_similar(
                item_key=self.item_key,
                max_results=5
            )
            for r in self.results:
                if (self.expected_title in r.title.lower() and 
                    r.score >= self.min_score):
                    self.passed = True
                    return
            self.error = "No matching similar document found"
            self.passed = False
        except Exception as e:
            self.error = str(e)
            self.passed = False


class IndexTest:
    """Test index status."""
    
    def __init__(self, min_papers=10):
        self.name = "Index Status"
        self.min_papers = min_papers
        self.passed = False
        self.error = None
        self.status = None
        
    async def run(self, client):
        """Run the test."""
        try:
            self.status = await client.get_index_status()
            if not self.status.ready:
                self.error = "Index not ready"
                self.passed = False
            elif self.status.indexed_papers < self.min_papers:
                self.error = f"Only {self.status.indexed_papers} papers (expected >= {self.min_papers})"
                self.passed = False
            else:
                self.passed = True
        except Exception as e:
            self.error = str(e)
            self.passed = False


# Helper functions for creating tests
def doc_test(name, query, expected_title, expected_authors=None, expected_year=None):
    """Create document retrieval test."""
    def check(results):
        for r in results:
            if expected_title.lower() in r.title.lower():
                if expected_authors:
                    if not r.authors:
                        continue
                    if not any(a.lower() in [x.lower() for x in r.authors] 
                               for a in expected_authors):
                        continue
                if expected_year and r.year != expected_year:
                    continue
                return True
        return False
    return Test(name, query, check)


def paragraph_test(name, query, expected_text, expected_page=None, expected_section=None):
    """Create paragraph retrieval test."""
    return ParagraphTest(name, query, expected_text, 
                        expected_page=expected_page, 
                        expected_section=expected_section)


def similar_test(name, item_key, expected_title, min_score=0.5):
    """Create similar document test."""
    return SimilarTest(name, item_key, expected_title, min_score)


def index_test(min_papers=10):
    """Create index status test."""
    return IndexTest(min_papers)


# Example tests - CUSTOMIZE THESE with your actual papers
EXAMPLE_TESTS = [
    # 1. Check index is ready
    index_test(min_papers=10),
    
    # 2. Document retrieval - customize with your papers
    doc_test(
        name="Find Attention paper by title",
        query="Attention Is All You Need",
        expected_title="Attention Is All You Need",
        expected_authors=["Vaswani"],
        expected_year=2017
    ),
    
    doc_test(
        name="Find Attention paper by concept",
        query="transformer neural network",
        expected_title="Attention Is All You Need"
    ),
    
    # 3. Paragraph retrieval - customize with actual content from your papers
    paragraph_test(
        name="Find self-attention paragraph",
        query="self-attention mechanism",
        expected_text="self-attention",
        expected_section="methods"
    ),
    
    paragraph_test(
        name="Find results paragraph on page 5",
        query="experimental results",
        expected_text="results show",
        expected_page=5
    ),
    
    # 4. Similar documents - customize with your item keys
    similar_test(
        name="Find similar to transformer paper",
        item_key="ABCD1234",  # Replace with actual 8-char Zotero key
        expected_title="BERT",
        min_score=0.5
    ),
]


async def run_tests(client, tests):
    """Run all tests and print results."""
    print("Running retrieval tests...")
    print("=" * 50)
    
    for test in tests:
        print(f"\n{test.name}")
        await test.run(client)
        status = "✓ PASS" if test.passed else "✗ FAIL"
        print(f"  {status}")
        if test.error:
            print(f"  Error: {test.error}")
        if hasattr(test, 'results') and test.results:
            print(f"  Results: {len(test.results)}")
        if hasattr(test, 'status') and test.status:
            print(f"  Papers: {test.status.indexed_papers}, "
                  f"Chunks: {test.status.total_chunks}")
    
    passed = sum(1 for t in tests if t.passed)
    total = len(tests)
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    return passed == total


async def main():
    """Main entry point."""
    print("ZotSeek Retrieval Performance Tests")
    print("=" * 50)
    print("\nBefore running:")
    print("- Start Zotero with ZotSeek plugin")
    print("- Enable AI Agent Access in ZotSeek settings")
    print("- Enable HTTP server in Zotero settings")
    print("- Index your library")
    print("- Edit EXAMPLE_TESTS in this file with your papers")
    print()
    
    try:
        async with ZotSeekClient() as client:
            success = await run_tests(client, EXAMPLE_TESTS)
            return 0 if success else 1
    except ConnectionError as e:
        print(f"Connection error: {e}")
        print("\nCheck that Zotero with ZotSeek is running.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

from mcp.server.fastmcp import FastMCP

from tools.web_tools import search_web
from tools.summarizer import summarize_text
from tools.notes import save_notes

mcp = FastMCP("ResearchAgent")

# ------------------------------------
# TOOL 1: Web Search
# ------------------------------------
@mcp.tool()
def web_search(query: str):
    """Search the web"""
    return search_web(query)

# ------------------------------------
# TOOL 2: Summarize
# ------------------------------------
@mcp.tool()
def summarize(content: str):
    """Summarize research content"""
    return summarize_text(content)

# ------------------------------------
# TOOL 3: Save Notes
# ------------------------------------
@mcp.tool()
def save(topic: str, content: str):
    """Save research notes"""
    return save_notes(topic, content)

# Run MCP server
if __name__ == "__main__":
    mcp.run()
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("CalculatorTools")

# -----------------------------
# TOOL 1: Add
# -----------------------------
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# -----------------------------
# TOOL 2: Multiply
# -----------------------------
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# -----------------------------
# TOOL 3: Square
# -----------------------------
@mcp.tool()
def square(x: int) -> int:
    """Square a number"""
    return x * x

# Run server
if __name__ == "__main__":
    mcp.run()
from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
import ollama

mcp = FastMCP("ResearchTools")

# -----------------------------------
# TOOL: WEB SEARCH
# -----------------------------------
@mcp.tool()
def web_search(query: str) -> str :
    results = []
    with DDGS() as ddgs:
        data = ddgs.text(query, max_results = 3)

        for r in data:
            results.append(
                f"""
                TITLE: {r.get('title')}

                BODY:
                {r.get('body')}

                URL:
                {r.get('href')}
                """
            )

            return "\n".join(results)
        
# -----------------------------------
# TOOL: SUMMARIZER
# -----------------------------------
@mcp.tool()
def summarize(content: str) -> str:
    prompt = f"""
            Summarize this content into concise bullet points.

            CONTENT:
            {content}
            """

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    mcp.run()
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize connection
            await session.initialize()

            # ---------------------------------------
            # CALL MCP TOOL: web_search
            # ---------------------------------------
            search_result = await session.call_tool(
                "web_search",
                {
                    "query": "Latest AI agent frameworks"
                }
            )

            print("\nSEARCH RESULTS:\n")
            print(search_result)

            # ---------------------------------------
            # CALL MCP TOOL: summarize
            # ---------------------------------------
            summary = await session.call_tool(
                "summarize",
                {
                    "content": str(search_result)
                }
            )

            print("\nSUMMARY:\n")
            print(summary)

            # ---------------------------------------
            # CALL MCP TOOL: save_notes
            # ---------------------------------------
            save_result = await session.call_tool(
                "save_notes",
                {
                    "topic": "AI Agents",
                    "content": str(summary)
                }
            )

            print("\nSAVE RESULT:\n")
            print(save_result)

asyncio.run(main())
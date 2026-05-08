import asyncio
import json
import ollama

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM_PROMPT = """
                You are an AI research agent.

                Available MCP tools:

                1. web_search
                Purpose:
                Search the internet.

                Arguments:
                {
                    "query": "search query"
                }

                2. summarize
                Purpose:
                Summarize content.

                Arguments:
                {
                    "content": "text"
                }

                IMPORTANT:

                When you need a tool:
                Return ONLY valid JSON.

                Format:
                {
                    "tool": "tool_name",
                    "arguments": {
                        ...
                    }
                }

                If no tool needed:
                Return:
                {
                "final_answer": "your response"
                }
                """

# -----------------------------------------------------
# Ask Ollama what tool to call
# -----------------------------------------------------
def ask_llm(messages):
    response = ollama.chat(
        model="llama3",
        messages = messages
    )

    return response["message"]["content"]


# -----------------------------------------------------
# MAIN AGENT LOOP
# -----------------------------------------------------
async def run_agent(user_query):
    server_params = StdioServerParameters(
        command = "python",
        args = ["server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ] 

            for step in range(5):
                print(f"\n===== AGENT STEP {step+1} =====\n")

                llm_response = ask_llm(messages)

                print("LLM RESPONSE:")
                print(llm_response)

                try:
                    action = json.loads(llm_response)
                    print(action)

                except:
                    print("Failed to parse JSON.")
                    break

                # -----------------------------------------
                # FINAL ANSWER
                # -----------------------------------------

                if "final_answer" in action:

                    print("\nFINAL ANSWER:\n")
                    print(action["final_answer"])
                    return
                
                # -----------------------------------------
                # TOOL CALL
                # -----------------------------------------

                tool_name = action["tool"]
                arguments = action["arguments"]

                print(f"\nCALLING TOOL: {tool_name}")
                print(arguments)

                tool_result = await session.call_tool(
                    tool_name,
                    arguments
                )

                print("\nTOOL RESULT:\n")
                print(tool_result)

                # Feed result back into LLM
                messages.append(
                    {
                        "role": "assistant",
                        "content": llm_response
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": f"""
                                    Tool result from {tool_name}:

                                    {tool_result}

                                    Now continue solving the task.
                                    """
                    }
                )


# -----------------------------------------------------
# RUN
# -----------------------------------------------------
if __name__ == "__main__":

    query = input("Enter research task: ")

    asyncio.run(run_agent(query))
# -----------------------------------
# Imports
# -----------------------------------

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser

# Your tools
from tools import (
    scrape_website,
    search_and_scrape,
    save_to_txt
)


# -----------------------------------
# Pydantic Models
# -----------------------------------

class LeadResponse(BaseModel):
    company: str
    contact_info: str
    email: str
    summary: str
    outreach_message: str
    tools_used: list[str]


class LeadResponseList(BaseModel):
    leads: list[LeadResponse]


# -----------------------------------
# LLM
# -----------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# -----------------------------------
# Parser
# -----------------------------------

parser = PydanticOutputParser(
    pydantic_object=LeadResponseList
)


# -----------------------------------
# System Prompt
# IMPORTANT:
# create_agent() expects STRING
# NOT ChatPromptTemplate
# -----------------------------------

system_prompt = f"""
You are a sales enablement assistant.

Your tasks:

1. Use the scrape_website tool to find exactly 5 local
small businesses in Vancouver, British Columbia.

2. Use the search_and_scrape tool to gather
detailed information.

3. Analyze businesses for IT services opportunities.

For each business provide:
- company
- contact_info
- email
- summary
- outreach_message
- tools_used

IMPORTANT:
- Return ONLY valid JSON
- No markdown
- No explanations
- No extra text

Output format:

{parser.get_format_instructions()}

After generating JSON,
use save_to_txt tool to save results.
"""


# -----------------------------------
# Tools
# -----------------------------------

tools = [
    scrape_website,
    search_and_scrape,
    save_to_txt
]


# -----------------------------------
# Create Modern Agent
# -----------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# -----------------------------------
# User Query
# -----------------------------------

query = """
Find and qualify exactly 5 local leads
in Vancouver, British Columbia for IT services.
"""


# -----------------------------------
# Invoke Agent
# -----------------------------------

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query
        }
    ]
})


# -----------------------------------
# Debug Output
# -----------------------------------

print("\nRAW RESPONSE:\n")
print(response)


# -----------------------------------
# Extract Final Message
# -----------------------------------

try:

    final_output = response["messages"][-1].content

    print("\nFINAL OUTPUT:\n")
    print(final_output)

    structured_response = parser.parse(final_output)

    print("\nPARSED RESPONSE:\n")
    print(
        structured_response.model_dump_json(
            indent=2
        )
    )

except Exception as e:

    print("\nParsing Error:\n", e)
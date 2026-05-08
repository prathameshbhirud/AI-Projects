import ollama
from server import add, multiply, square

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Available tools:
1. add(a, b)
2. multiply(a, b)
3. square(x)

When solving math:
- Think step-by-step
- Explain which tool should be called
"""

def route_tool(question: str):

    if "multiply" in question or "*" in question:
        return multiply(45, 12)

    if "square" in question:
        return square(5)

    if "add" in question or "+" in question:
        return add(10, 20)

    return "No tool matched"

def ask_ai(question):
    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]

if __name__ == "__main__":
    question = input("Ask something: ")

    result = route_tool(question)
    print("Tool Result:", result)

    answer = ask_ai(question)

    print("\nAI Response:\n")
    print(answer)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

rewrite_prompt = ChatPromptTemplate.from_template("""
                Rewrite the user question to improve retrieval.

                Original question:
                {question}

                Focus on:
                - keywords
                - entities
                - technical terminology
                - concise retrieval phrasing

                Return ONLY the rewritten query.
                """)

def rewrite_node(state):

    retries = state["retries"] + 1

    chain = rewrite_prompt | llm

    rewritten = chain.invoke({
        "question": state["question"]
    })

    return {
        "rewritten_question": rewritten.content,
        "retries": retries
    }
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant.

            Answer ONLY using the provided context.

            If the answer is not in context, say:
            "I don't have enough information."

            Context:
            {context}

            Question:
            {question}
            """)

def generate_node(state):

    context = "\n\n".join(state["documents"])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })

    return {
        "answer": response.content
    }
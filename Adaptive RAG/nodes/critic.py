from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

critic_prompt = ChatPromptTemplate.from_template("""
                You are a strict evaluator.

                Determine whether the answer is fully grounded
                in the provided context.

                Context:
                {context}

                Question:
                {question}

                Answer:
                {answer}

                Respond ONLY in JSON:

                {{
                    "grounded": true/false,
                    "reason": "short explanation"
                }}
                """)

def critic_node(state):

    context = "\n\n".join(state["documents"])

    chain = critic_prompt | llm

    result = chain.invoke({
        "context": context,
        "question": state["question"],
        "answer": state["answer"]
    })

    parsed = json.loads(result.content)

    return {
        "grounded": parsed["grounded"],
        "critique": parsed["reason"]
    }
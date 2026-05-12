from langgraph.graph import StateGraph, END

from graph import GraphState

from nodes.retrieve import retrieve_node
from nodes.generate import generate_node
from nodes.critic import critic_node
from nodes.rewrite import rewrite_node

MAX_RETRIES = 2

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("critic", critic_node)
workflow.add_node("rewrite", rewrite_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critic")

def route_after_critic(state):

    if state["grounded"]:
        return END

    if state["retries"] >= MAX_RETRIES:
        return END

    return "rewrite"

workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        END: END,
        "rewrite": "rewrite"
    }
)

workflow.add_edge("rewrite", "retrieve")

app = workflow.compile()

if __name__ == "__main__":

    question = input("Question: ")

    result = app.invoke({
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "answer": "",
        "critique": "",
        "grounded": False,
        "retries": 0
    })

    print("\nANSWER:")
    print(result["answer"])

    print("\nCRITIQUE:")
    print(result["critique"])
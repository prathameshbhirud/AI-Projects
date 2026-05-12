from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[str]
    answer: str
    critique: str
    grounded: bool
    retries: int
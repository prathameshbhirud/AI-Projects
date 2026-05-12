from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def retrieve_node(state):
    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    docs = retriever.invoke(query)

    return {
        "documents": [d.page_content for d in docs]
    }
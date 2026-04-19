from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# initialize embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    request_timeout = 300
)

# initialize LLM with settings
llm = Ollama(
    model="gemma:2b",
    request_timeout=300,
    max_tokens = 100,
    context_window = 4096,
    temperature=0.1         # lower the temperature, lower creativity and more factual answers
)

# global config
Settings.embed_model = embed_model
Settings.llm = llm


def load_and_index_documents(data_dir="data"):
    """Load documents and create vector index"""

    # check if directory exists
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and add you PDF files to it.")
    
    # load from folder
    docs = SimpleDirectoryReader(data_dir).load_data()

    if not docs:
        raise ValueError(f"No documents found in '{data_dir}'")
    
    # create vector index from document
    index = VectorStoreIndex.from_documents(docs, embed_model = embed_model)

    return index

def create_query_engine(index, similarity_top_k=3):
    """Create query index with specified retrieval parameters"""

    query_engine = index.as_query_engine(
        llm = llm,
        similarity_top_k = similarity_top_k,    # number of relevant chunks to retrieve
        response_mode = "compact"
    )

    return query_engine


def test_rag_system():
    """Test RAG with sample queries"""

    try:
        # load documents and create index
        index = load_and_index_documents()

        # create query engine
        query_engine = create_query_engine(index)

        # test queries
        test_queries = [
            "Summarize document in 5 lines",
            "what are the main topics covered in the document?"
        ]

        print("RAG Test results")
        print("-" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 30)

            try:
                response = query_engine.query(query)
                print(f"Response: {response}")
                print(f"Status: SUCCESS")
            except Exception as e:
                print(f"Error: {str(e)}")
                print(f"Status: FAILED")

            print("-" * 50)

        return True
    
    except Exception as ex:
        print(f"System error : {str(ex)}")
        return False
    

# Main execution
if __name__ == "__main__":
    print("Starting RAG testing...")

    # Test Complete system
    success = test_rag_system()

    if success:
        print("\nRAG system is working correctly!")
        print("You can now use the query_engine to ask questions about your documents.")
    else:
        print("\nRAG system test failed. Check the error messages above.")


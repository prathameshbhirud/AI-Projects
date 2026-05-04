from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# specify model to be used
llm = Ollama(model="gemma:2b")

# web search tool
search = DuckDuckGoSearchRun()

# Prompt template - instruction for LLM
prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. You must answer user's questions based only on following search results. 
    If the search results are empty or do not contain any answer, say 'I could not find any information on that.'

    Search Results: 
    {context}

    Question: 
    {question}
    """
)

# chain for RAG
chain = (
    RunnablePassthrough.assign(
        # "context" is a new key we add to the dictionary.
        # Its value is the *output* of running the 'search' tool
        # with the original 'question' as input.
        context = lambda x: search.run(x["question"])
    )
    | prompt  # The dictionary (now with 'context' and 'question') is "piped" into the prompt
    | llm     # The formatted prompt is "piped" into the LLM
)

print("Hello! I'm a real-time AI assistant. What's new?")
while True:
    try:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        print("Thinking...")
        
        # RAG proces invoked
        response = chain.invoke({"question": user_query})
        
        print(f"{response}")

    except Exception as e:
        print(f"An error occurred: {e}")
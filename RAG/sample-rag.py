import ollama

# reading context file / dataset
dataset = []
with open('cat-facts.txt', 'r', encoding="utf-8") as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')


# implementing Retrieval system

# Embedding and chunking
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# each element in VECTOR_DB is a tuple (chunk, embedding) type
VECTOR_DB = []

def add_chunk_to_vector_db(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))


for i, chunk in enumerate(dataset):
    add_chunk_to_vector_db(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

def consine_similarity(a, b):
    dot_product = sum([ x * y for x, y in zip(a,b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([y ** 2 for y in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


def retrieve(query, top = 3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    # temp list to store tuple of chunk, embedding
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = consine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    # sort similarities in descending order, higher means more relevant chunk
    similarities.sort(key = lambda x: x[1], reverse=True)
    return similarities[:top]


# Chatbot implementation

input_query = input('Ask me a quetion: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge.')

for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

context_text = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])

instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_text}
"""

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        { 'role': 'system', 'content' : instruction_prompt},
        { 'role': 'user', 'content' : input_query},
    ],
    stream=True
)

# Print response from chatbot in real time
print('Chatbot response.')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

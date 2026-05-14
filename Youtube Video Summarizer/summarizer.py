from ollama import chat

def summarize(text: str):
    prompt = f'''
    Summarize the following YouTube transcript.

    Focus on:
    - main ideas
    - important insights
    - key technical concepts

    Transcript:
    {text}
    '''

    response = chat(
        model="qwen2.5:3b",
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]
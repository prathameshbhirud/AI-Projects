import ollama

def summarize_text(content: str):

    prompt = f"""
                Summarize the following research content.

                CONTENT:
                {content}

                Provide:
                - key insights
                - important trends
                - concise bullets
              """

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
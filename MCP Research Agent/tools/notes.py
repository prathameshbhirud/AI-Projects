from datetime import datetime

def save_notes(topic: str, content: str):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"notes/{topic}_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename
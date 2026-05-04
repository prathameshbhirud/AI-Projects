import json

data = []

with open("train_data.jsonl") as f:
    for line in f:
        item = json.loads(line)

        data.append({
            "prompt": item["prompt"],
            "response": json.dumps(item["response"])
        })

with open("train.json", "w") as f:
    json.dump(data, f, indent=2)
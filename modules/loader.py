import json

def load_prompts(path):
    prompts = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data["prompt"])

    return prompts

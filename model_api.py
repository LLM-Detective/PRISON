import json
import openai
from openai import OpenAI

def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_model(prompt: str, model_name: str, config_path="config.json") -> str:
    config = load_config(config_path)
    model_name = model_name.lower().strip()

    if model_name == "deepseek-v3":
        deepseek_cfg = config["deepseek"]
        client = OpenAI(
            api_key=deepseek_cfg["api_key"],
            base_url=deepseek_cfg["base_url"]
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content

    else:
        v3_cfg = config["v3"]

        openai.api_key = v3_cfg["api_key"]
        openai.base_url = v3_cfg["base_url"]
        openai.default_headers = v3_cfg["default_headers"]

        model_mapping = {
            "gpt-4o": "gpt-4o",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gemini-2-flash": "gemini-2.0-flash",
            "gemini-1.5-flash": "gemini-1.5-flash",
            "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
            "qwen-max": "qwen-max-latest",
            "qwen2.5": "Qwen2.5-72B-Instruct"
        }

        model = model_mapping.get(model_name)
        if not model:
            raise ValueError(f"Unsupported model: {model_name}")

        messages = [
            {"role": "user", "content": prompt}
        ]

        completion = openai.chat.completions.create(
            model=model,
            messages=messages
        )

        return completion.choices[0].message.content

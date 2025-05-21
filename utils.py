import json
import re

class SceneManager:
    def __init__(self, single_prompt_path: str, dialogue_prompt_path: str, scene_data_path: str):
        self.single_prompt = self._load_template(single_prompt_path)
        self.dialogue_prompt = self._load_template(dialogue_prompt_path)
        self.scene_data = self._load_scene_data(scene_data_path)

    def _load_template(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _load_scene_data(self, filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_scene_data(self) -> dict:
        return self.scene_data

    def fill_prompt_single(self, data: dict, intention: str) -> str:
        filled_data = data.copy()
        if intention == "with":
            filled_data["intention_section"] = f"Here is the intention behind your next actions:\n{intention}"
        else:
            filled_data["intention_section"] = ""
        return self.single_prompt.format(**filled_data)

    def fill_prompt_dialogue(self, character: dict, history_dialogue: str, intention: str) -> str:
        filled_data = character.copy()
        filled_data["history_dialogue"] = history_dialogue
        if intention == "with":
            filled_data["intention_section"] = f"Here is the intention behind your next actions:\n{intention}"
        else:
            filled_data["intention_section"] = ""
        return self.dialogue_prompt.format(**filled_data)

    def extract_parts(self, text: str) -> dict:
        thought_match = re.search(r'### THOUGHT:\s*(.*?)\s*### RESPONSE:', text, re.DOTALL)
        response_match = re.search(r'### RESPONSE:\s*(.*)', text, re.DOTALL)
        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "response": response_match.group(1).strip() if response_match else ""
        }

class AnnotateManager:
    def __init__(self, prompt_path: str):
        self.prompt = self._load_template(prompt_path)

    def _load_template(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def fill_prompt(self, data: dict) -> str:
        filled_data = data.copy()
        return self.prompt.format(**filled_data)

    def extract_json_from_output(self, text: str) -> dict:
        # print(text)
        blocks = re.split(r"\n\s*---+\s*\n", text.strip())
        results = []
        for block in blocks:
            if not block.strip():
                continue
            entry = {"sentence": "", "tags": {}}
            lines = block.strip().splitlines()
            for line in lines:
                if line.startswith("Sentence:"):
                    entry["sentence"] = line.replace("Sentence:", "").strip()
                else:
                    match = re.search(r"([\w\s\-]+):\s*(null|score=1\s*\|\s*explanation=(.+))", line.strip())
                    if match:
                        tag = match.group(1)
                        if match.group(2) == "null":
                            entry["tags"][tag] = None
                        else:
                            explanation = match.group(3).strip()
                            entry["tags"][tag] = {"score": 1, "explanation": explanation}         
            results.append(entry)
        return results

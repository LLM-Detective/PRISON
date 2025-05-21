import os
import json
import argparse
from copy import deepcopy

from utils import SceneManager
from model_api import call_model

OUTPUT_DIR = "scene"

def main(model_name: str, intention: str):
    scene_manager = SceneManager(
        single_prompt_path="prompt/prompt_single.txt",
        dialogue_prompt_path="prompt/prompt_dialogue.txt",
        scene_data_path="data/criminal_scene.json"
    )

    scene_data = scene_manager.get_scene_data()

    output_path = f"{OUTPUT_DIR}/{intention}/{model_name}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dialogue_turns = 5 # default

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    else:
        existing_results = {"single": {}, "dialogue": {}}

    output = deepcopy(existing_results)

    # Process single-character scenes
    for scene_id, characters in scene_data.get("single", {}).items():
        if scene_id in existing_results.get("single", {}):
            print(f"[Skipped] Single: {scene_id} already processed.")
            continue

        print(f"[Processing] Single scene: {scene_id}")
        results = {}
        for name, info in characters.items():
            if name == "source":
                continue
            info["name"] = name
            info["intention"] = intention
            prompt = scene_manager.fill_prompt_single(info, intention)
            try:
                reply = call_model(prompt, model_name)
                results[name] = scene_manager.extract_parts(reply)
                print(f"{name} success.")
            except Exception as e:
                print(f"{name} error: {e}")
                results[name] = {"thought": "", "response": "", "error": str(e)}

        output["single"][scene_id] = characters
        output["single"][scene_id]["results"] = results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # Process dialogue scenes
    for scene_id, characters in scene_data.get("dialogue", {}).items():
        if scene_id in existing_results.get("dialogue", {}):
            print(f"[Skipped] Dialogue: {scene_id} already processed.")
            continue

        print(f"[Processing] Dialogue scene: {scene_id}")
        names = [k for k in characters if k not in ("origin", "source")]
        if len(names) != 2:
            print(f"Invalid number of characters in {scene_id}")
            continue

        first, second = (names[0], names[1]) if "dialogue" in characters[names[0]] else (names[1], names[0])
        characters[first]["name"] = first
        characters[first]["intention"] = intention
        characters[second]["name"] = second
        characters[second]["intention"] = intention

        history_dialogue = ""
        all_results = {}

        for turn in range(1, dialogue_turns + 1):
            all_results[turn] = {}

            if turn == 1:
                history_dialogue += f"{first}: {characters[first]['dialogue']}\n"
                all_results[turn][first] = {
                    "thought": "",
                    "response": characters[first]["dialogue"]
                }
            else:
                prompt = scene_manager.fill_prompt_dialogue(characters[first], history_dialogue, intention)
                try:
                    reply = call_model(prompt, model_name)
                    parsed = scene_manager.extract_parts(reply)
                    all_results[turn][first] = parsed
                    history_dialogue += f"{first}: {parsed['response']}\n"
                except Exception as e:
                    all_results[turn][first] = {"thought": "", "response": "", "error": str(e)}
                    break

            prompt =scene_manager.fill_prompt_dialogue(characters[second], history_dialogue, intention)
            try:
                reply = call_model(prompt, model_name)
                parsed = scene_manager.extract_parts(reply)
                all_results[turn][second] = parsed
                history_dialogue += f"{second}: {parsed['response']}\n"
            except Exception as e:
                all_results[turn][second] = {"thought": "", "response": "", "error": str(e)}
                break

        output["dialogue"][scene_id] = characters
        output["dialogue"][scene_id]["results"] = all_results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"All results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model to use")
    parser.add_argument("intention", type=str, help="The intention description to inject")
    args = parser.parse_args()

    main(args.model_name, args.intention)

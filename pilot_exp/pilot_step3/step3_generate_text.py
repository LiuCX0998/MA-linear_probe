import json
import os
from typing import Dict, List

from pilot.config import STEP2_DIR, STEP3_DIR


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_object_type(obj_type: str) -> str:
    """
    把 AI2-THOR 的 objectType 转成更自然的英文短语。
    这里只做最小处理：
    - CounterTop -> countertop
    - CoffeeMachine -> coffee machine
    - SaltShaker -> salt shaker
    - 其他 CamelCase 也尽量拆开
    """
    if not obj_type:
        return ""

    chars = []
    for i, ch in enumerate(obj_type):
        if i > 0 and ch.isupper() and obj_type[i - 1].islower():
            chars.append(" ")
        chars.append(ch)

    return "".join(chars).strip().lower()


def choose_article(noun_phrase: str) -> str:
    """
    非严格英语处理：
    根据首字母简单决定 a / an。
    """
    if not noun_phrase:
        return "a"

    first = noun_phrase[0].lower()
    if first in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


def make_np(obj_type: str, use_definite: bool = True) -> str:
    """
    生成名词短语：
    - the fork
    - the drawer
    这里默认统一用 the，最简单也最自然。
    """
    phrase = normalize_object_type(obj_type)
    if use_definite:
        return f"the {phrase}"
    article = choose_article(phrase)
    return f"{article} {phrase}"


def relation_to_sentence(rel: Dict) -> str:
    """
    把一条关系转成一句英文。
    当前只处理 in / on。
    """
    subject_type = rel["subject_type"]
    object_type = rel["object_type"]
    relation = rel["relation"]

    subject_np = make_np(subject_type, use_definite=True)
    object_np = make_np(object_type, use_definite=True)

    if relation == "in":
        return f"{subject_np.capitalize()} is in {object_np}."
    elif relation == "on":
        return f"{subject_np.capitalize()} is on {object_np}."
    else:
        return f"{subject_np.capitalize()} is related to {object_np} by {relation}."


def build_text_output(scene_relations: Dict) -> Dict:
    """
    输入一个 scene 的 step2 relations JSON，
    输出 step3 的文本 JSON。
    """
    scene = scene_relations["scene"]
    relations = scene_relations.get("relations", [])

    sentences: List[str] = []
    text_records: List[Dict] = []

    for idx, rel in enumerate(relations):
        sentence = relation_to_sentence(rel)
        sentences.append(sentence)

        text_records.append(
            {
                "relation_index": idx,
                "subject_id": rel["subject_id"],
                "subject_type": rel["subject_type"],
                "relation": rel["relation"],
                "object_id": rel["object_id"],
                "object_type": rel["object_type"],
                "sentence": sentence,
            }
        )

    paragraph = " ".join(sentences)

    return {
        "scene": scene,
        "num_sentences": len(sentences),
        "sentences": sentences,
        "paragraph": paragraph,
        "text_records": text_records,
    }


def main() -> None:
    ensure_dir(STEP3_DIR)

    for filename in os.listdir(STEP2_DIR):
        if not filename.endswith(".json"):
            continue

        in_path = os.path.join(STEP2_DIR, filename)
        scene_relations = load_json(in_path)

        out_data = build_text_output(scene_relations)

        out_filename = filename.replace("_relations.json", "_text.json")
        out_path = os.path.join(STEP3_DIR, out_filename)

        save_json(out_path, out_data)

        print(
            f"{out_data['scene']}: "
            f"{out_data['num_sentences']} sentences saved to {out_filename}"
        )


if __name__ == "__main__":
    main()
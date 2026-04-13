import json
import os
from typing import Dict, List

from pilot_2.config import STEP2_DIR, STEP3_DIR


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
    生成名词短语。
    """
    phrase = normalize_object_type(obj_type)
    if use_definite:
        return f"the {phrase}"
    article = choose_article(phrase)
    return f"{article} {phrase}"


def should_use_relation_for_text(rel: Dict) -> bool:
    """
    决定某条关系是否进入 step3 文本。
    当前策略：
    - 必须 candidate_for_text = True
    - 必须 verbalizable = True
    - 默认排除反向关系，避免文本重复
    """
    if not rel.get("candidate_for_text", True):
        return False
    if not rel.get("verbalizable", True):
        return False
    return True


def relation_to_sentence(rel: Dict) -> str:
    """
    把一条关系转成一句更自然的英文。
    支持 step2 当前输出的关系类型。
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
    elif relation == "contains":
        return f"{subject_np.capitalize()} contains {object_np}."
    elif relation == "supports":
        return f"{subject_np.capitalize()} supports {object_np}."
    elif relation == "left_of":
        return f"{subject_np.capitalize()} is to the left of {object_np}."
    elif relation == "right_of":
        return f"{subject_np.capitalize()} is to the right of {object_np}."
    elif relation == "above":
        return f"{subject_np.capitalize()} is above {object_np}."
    elif relation == "below":
        return f"{subject_np.capitalize()} is below {object_np}."
    elif relation == "near":
        return f"{subject_np.capitalize()} is near {object_np}."
    else:
        return f"{subject_np.capitalize()} is related to {object_np} by {relation}."


def build_text_output(scene_relations: Dict) -> Dict:
    """
    输入一个 scene 的 step2 relations JSON，
    输出 step3 的文本 JSON。
    """
    scene = scene_relations["scene"]
    all_relations = scene_relations.get("relations", [])

    selected_relations = [
        rel for rel in all_relations
        if should_use_relation_for_text(rel)
    ]

    sentences: List[str] = []
    text_records: List[Dict] = []

    for idx, rel in enumerate(selected_relations):
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
                "relation_family": rel.get("relation_family"),
                "is_inverse_relation": rel.get("is_inverse_relation", False),
                "derived_from_relation": rel.get("derived_from_relation"),
                "evidence": rel.get("evidence", {}),
                "sentence": sentence,
            }
        )

    paragraph = " ".join(sentences)

    return {
        "scene": scene,
        "num_input_relations": len(all_relations),
        "num_selected_relations": len(selected_relations),
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
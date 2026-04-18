import json
import os
from typing import Dict, List
from collections import defaultdict

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

    # Convert AI2-THOR objectType into a more natural English phrase.
    if not obj_type:
        return ""

    chars = []
    for i, ch in enumerate(obj_type):
        if i > 0 and ch.isupper() and obj_type[i - 1].islower():
            chars.append(" ")
        chars.append(ch)

    return "".join(chars).strip().lower()


def should_use_relation_for_text(rel: Dict) -> bool:

    # Determine whether a relation should be included in Step 3 text.
    # Current policy:
    # - candidate_for_text must be True
    # - verbalizable must be True
    if not rel.get("candidate_for_text", True):
        return False
    if not rel.get("verbalizable", True):
        return False
    return True

def build_entity_alias_map(relations: List[Dict]) -> Dict[str, str]:

    # Assign a readable alias to each unique entity ID within the scene.
    # Use the full subject_id / object_id as the key to avoid confusion between same-type entities.
    ids_by_type = defaultdict(set)

    for rel in relations:
        subject_id = rel["subject_id"]
        subject_type = rel["subject_type"]
        object_id = rel["object_id"]
        object_type = rel["object_type"]

        ids_by_type[subject_type].add(subject_id)
        ids_by_type[object_type].add(object_id)

    alias_map: Dict[str, str] = {}

    for obj_type, full_ids in ids_by_type.items():
        base = normalize_object_type(obj_type).replace(" ", "_")
        for idx, full_id in enumerate(sorted(full_ids), start=1):
            alias_map[full_id] = f"{base}_{idx}"

    return alias_map

def relation_to_sentence(rel: Dict, alias_map: Dict[str, str]) -> str:

    # Convert a relation into an English sentence.
    # Use alias_map to distinguish different instances of the same type.
    subject_alias = alias_map[rel["subject_id"]]
    object_alias = alias_map[rel["object_id"]]
    relation = rel["relation"]

    if relation == "in":
        return f"{subject_alias} is in {object_alias}."
    elif relation == "on":
        return f"{subject_alias} is on {object_alias}."
    elif relation == "contains":
        return f"{subject_alias} contains {object_alias}."
    elif relation == "supports":
        return f"{subject_alias} supports {object_alias}."
    elif relation == "left_of":
        return f"{subject_alias} is to the left of {object_alias}."
    elif relation == "right_of":
        return f"{subject_alias} is to the right of {object_alias}."
    elif relation == "above":
        return f"{subject_alias} is above {object_alias}."
    elif relation == "below":
        return f"{subject_alias} is below {object_alias}."
    elif relation == "near":
        return f"{subject_alias} is near {object_alias}."
    else:
        return f"{subject_alias} is related to {object_alias} by {relation}."


def build_text_output(scene_relations: Dict) -> Dict:

    scene = scene_relations["scene"]
    all_relations = scene_relations.get("relations", [])

    selected_relations = [
        rel for rel in all_relations
        if should_use_relation_for_text(rel)
    ]

    alias_map = build_entity_alias_map(selected_relations)

    sentences: List[str] = []
    text_records: List[Dict] = []

    for idx, rel in enumerate(selected_relations):
        sentence = relation_to_sentence(rel, alias_map)
        sentences.append(sentence)

        text_records.append(
            {
                "relation_index": idx,
                "subject_id": rel["subject_id"],
                "subject_type": rel["subject_type"],
                "subject_alias": alias_map[rel["subject_id"]],
                "relation": rel["relation"],
                "object_id": rel["object_id"],
                "object_type": rel["object_type"],
                "object_alias": alias_map[rel["object_id"]],
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
        "entity_alias_map": alias_map,
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
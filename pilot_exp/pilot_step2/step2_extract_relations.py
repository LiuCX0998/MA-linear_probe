import json
import os

from pilot.config import STEP1_DIR, STEP2_DIR


SURFACE_TYPES = {
    "CounterTop",
    "TableTop",
    "Desk",
    "Shelf",
    "Sink",
    "StoveBurner"
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_object_index(objects):
    """objectId -> object record"""
    return {obj["objectId"]: obj for obj in objects}


def infer_relation(child_obj, parent_obj):
    """
    根据 parent 的类型判断是 in 还是 on
    """
    parent_type = parent_obj["objectType"]
    if parent_type in SURFACE_TYPES:
        return "on"
    else:
        return "in"


def build_relations(scene_data):
    objects = scene_data["objects"]
    obj_index = build_object_index(objects)

    relations = []

    for obj in objects:
        child_id = obj["objectId"]
        child_type = obj["objectType"]

        for parent_id in obj["parentReceptacles"]:
            if parent_id not in obj_index:
                continue  # 只保留 selected objects 内部关系

            parent_obj = obj_index[parent_id]
            parent_type = parent_obj["objectType"]

            relation = infer_relation(obj, parent_obj)

            relations.append({
                "subject_id": child_id,
                "subject_type": child_type,
                "relation": relation,
                "object_id": parent_id,
                "object_type": parent_type
            })

    return relations


def main():
    ensure_dir(STEP2_DIR)

    for filename in os.listdir(STEP1_DIR):
        if not filename.endswith(".json"):
            continue

        in_path = os.path.join(STEP1_DIR, filename)
        scene_data = load_json(in_path)

        relations = build_relations(scene_data)

        out_data = {
            "scene": scene_data["scene"],
            "num_relations": len(relations),
            "relations": relations
        }

        out_filename = filename.replace(".json", "_relations.json")
        out_path = os.path.join(STEP2_DIR, out_filename)

        save_json(out_path, out_data)

        print(f"{scene_data['scene']}: {len(relations)} relations saved.")


if __name__ == "__main__":
    main()
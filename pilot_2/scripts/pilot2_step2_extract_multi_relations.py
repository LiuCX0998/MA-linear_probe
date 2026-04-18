import json
import math
import os
import random

from pilot_2.config import STEP1_DIR, STEP2_DIR

SURFACE_TYPES = {
    "CounterTop",
    "TableTop",
    "Desk",
    "Shelf",
    "Sink",
    "StoveBurner"
}

STRUCTURAL_TYPES = {
    "Cabinet",
    "Drawer",
    "CounterTop",
    "Shelf",
    "Sink",
    "StoveBurner",
    "Desk",
    "TableTop",
    "Fridge",
    "Microwave",
    "CoffeeMachine",
    "Toaster",
    "Safe",
}

INVERSE_RELATION_MAP = {
    "in": "contains",
    "on": "supports",
    "left_of": "right_of",
    "above": "below",
}

# Thresholds for geometric relations; can be tuned in future iterations.
LEFT_MARGIN = 0.05
ABOVE_Y_MARGIN = 0.03
NEAR_THRESHOLD = 0.50


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


def infer_topological_relation(child_obj, parent_obj):

    # Determine whether the relation is "in" or "on" based on the parent type.
    parent_type = parent_obj["objectType"]
    if parent_type in SURFACE_TYPES:
        return "on"
    else:
        return "in"


def get_extent(obj):

    # Compute the spatial extent occupied by the object based on bbox.center and bbox.size.
    # If the bbox is missing, fall back to a degenerate case using position with size = 0.
    bbox = obj.get("bbox", {}) or {}
    center = bbox.get("center", {}) or obj.get("position", {}) or {}
    size = bbox.get("size", {}) or {}

    cx = float(center.get("x", 0.0))
    cy = float(center.get("y", 0.0))
    cz = float(center.get("z", 0.0))

    sx = float(size.get("x", 0.0))
    sy = float(size.get("y", 0.0))
    sz = float(size.get("z", 0.0))

    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": cz - sz / 2.0,
        "z_max": cz + sz / 2.0,
    }


def overlap_1d(a_min, a_max, b_min, b_max):
    return min(a_max, b_max) - max(a_min, b_min)


def has_overlap_1d(a_min, a_max, b_min, b_max, margin=0.0):
    return overlap_1d(a_min, a_max, b_min, b_max) > margin


def horizontal_distance(obj1, obj2):

    # Compute the Euclidean distance in the x–z plane.
    p1 = obj1.get("position", {}) or {}
    p2 = obj2.get("position", {}) or {}

    x1 = float(p1.get("x", 0.0))
    z1 = float(p1.get("z", 0.0))
    x2 = float(p2.get("x", 0.0))
    z2 = float(p2.get("z", 0.0))

    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


def is_left_of(obj_a, obj_b, margin=LEFT_MARGIN):

    #A is to the left of B:
    # determined by separation along the x-axis using bbox intervals.
    ea = get_extent(obj_a)
    eb = get_extent(obj_b)
    return ea["x_max"] < eb["x_min"] - margin


def is_above(obj_a, obj_b, y_margin=ABOVE_Y_MARGIN):

    # A is above B:
    # 1) The bottom of A is higher than the top of B
    # 2) And their projections overlap on the x/z plane
    ea = get_extent(obj_a)
    eb = get_extent(obj_b)

    x_overlap = has_overlap_1d(ea["x_min"], ea["x_max"], eb["x_min"], eb["x_max"])
    z_overlap = has_overlap_1d(ea["z_min"], ea["z_max"], eb["z_min"], eb["z_max"])
    y_clear = ea["y_min"] > eb["y_max"] + y_margin

    return x_overlap and z_overlap and y_clear


def is_near(obj_a, obj_b, threshold=NEAR_THRESHOLD):

    # A is near B: currently using a simple horizontal distance threshold.
    return horizontal_distance(obj_a, obj_b) <= threshold


def make_relation_record(
    subject_obj,
    relation,
    object_obj,
    family,
    evidence=None,
    candidate_for_text=True,
    verbalizable=True,
):
    return {
        "subject_id": subject_obj["objectId"],
        "subject_type": subject_obj["objectType"],
        "relation": relation,
        "object_id": object_obj["objectId"],
        "object_type": object_obj["objectType"],
        "relation_family": family,
        "candidate_for_text": candidate_for_text,
        "verbalizable": verbalizable,
        "evidence": evidence or {},
    }

def is_structural_object(obj):

    # Determine whether an object belongs to a large, structural category.
    # Used to filter out some geometrically valid but less textually meaningful relations.
    return obj.get("objectType") in STRUCTURAL_TYPES


def should_keep_left_of(obj_a, obj_b):

    # First-pass filtering rule for "left_of":
    # - If both objects are structural, discard the relation
    # - Otherwise, keep it

    if is_structural_object(obj_a) and is_structural_object(obj_b):
        return False
    return True


def should_keep_above(obj_a, obj_b):

    # First-pass filtering rule for "above":
    # - If both objects have exactly the same objectType and are structural (e.g., Drawer above Drawer),
    # filter it out
    # - If both objects are structural, also filter it out
    same_type = obj_a.get("objectType") == obj_b.get("objectType")
    if same_type and is_structural_object(obj_a) and is_structural_object(obj_b):
        return False

    if is_structural_object(obj_a) and is_structural_object(obj_b):
        return False

    return True

def should_keep_near(obj_a, obj_b):

    ## Filtering rules for "near":
    # 1) Do not keep structural object -> structural object
    # 2) Do not keep structural object -> small object
    # 3) Allow small object -> structural object
    # 4) Allow small object -> small object
    #
    # This helps avoid cases such as:
    # - Drawer near Spoon
    # - Cabinet near Drawer
    # which are geometrically valid but linguistically unnatural.

    a_structural = is_structural_object(obj_a)
    b_structural = is_structural_object(obj_b)

    if a_structural and b_structural:
        return False

    if a_structural and not b_structural:
        return False

    return True

def should_mark_verbalizable(rel):

    # Tag relations as verbalizable / candidate_for_text.
    relation = rel["relation"]
    subj_type = rel["subject_type"]
    obj_type = rel["object_type"]

    subj_structural = subj_type in STRUCTURAL_TYPES
    obj_structural = obj_type in STRUCTURAL_TYPES

    if rel["relation_family"] == "topological":
        return True

    if relation in {"left_of", "above"} and subj_structural and obj_structural:
        return False

    return True

def build_topological_relations(objects):

    # Recover "in" / "on" relations based on parentReceptacles.
    obj_index = build_object_index(objects)
    relations = []

    for obj in objects:
        for parent_id in obj.get("parentReceptacles", []):
            if parent_id not in obj_index:
                continue # Keep only relations among the selected objects.

            parent_obj = obj_index[parent_id]
            relation = infer_topological_relation(obj, parent_obj)

            relations.append(
                make_relation_record(
                    subject_obj=obj,
                    relation=relation,
                    object_obj=parent_obj,
                    family="topological"
                )
            )

    return relations


def build_geometric_relations(objects):

    # 1) Apply first-pass filtering to left_of / above
    # 2) Do not keep "near" bidirectionally (only keep i < j)
    # 3) If a stronger relation exists (in/on/left_of/above), discard "near"
    # 4) Add directional constraint for "near": structural objects cannot be subjects
    # 5) Add evidence to geometric relations
    relations = []

    topo_pairs = set()
    obj_index = build_object_index(objects)

    for obj in objects:
        for parent_id in obj.get("parentReceptacles", []):
            if parent_id in obj_index:
                topo_pairs.add((obj["objectId"], parent_id))

    n = len(objects)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            obj_a = objects[i]
            obj_b = objects[j]

            a_id = obj_a["objectId"]
            b_id = obj_b["objectId"]

            # If a topological relation (A -> B) already exists, skip geometric relations in that direction.
            if (a_id, b_id) in topo_pairs:
                continue

            has_strong_relation = False

            # ---------- left_of ----------
            if is_left_of(obj_a, obj_b) and should_keep_left_of(obj_a, obj_b):
                ea = get_extent(obj_a)
                eb = get_extent(obj_b)
                x_gap = eb["x_min"] - ea["x_max"]

                rel = make_relation_record(
                    subject_obj=obj_a,
                    relation="left_of",
                    object_obj=obj_b,
                    family="geometric",
                    evidence={
                        "rule": "x_separation",
                        "x_gap": x_gap,
                        "margin": LEFT_MARGIN,
                    },
                )
                relations.append(rel)
                has_strong_relation = True

            # ---------- above ----------
            if is_above(obj_a, obj_b) and should_keep_above(obj_a, obj_b):
                ea = get_extent(obj_a)
                eb = get_extent(obj_b)

                x_overlap = overlap_1d(ea["x_min"], ea["x_max"], eb["x_min"], eb["x_max"])
                z_overlap = overlap_1d(ea["z_min"], ea["z_max"], eb["z_min"], eb["z_max"])
                y_clearance = ea["y_min"] - eb["y_max"]

                rel = make_relation_record(
                    subject_obj=obj_a,
                    relation="above",
                    object_obj=obj_b,
                    family="geometric",
                    evidence={
                        "rule": "vertical_clearance_with_xz_overlap",
                        "y_clearance": y_clearance,
                        "x_overlap": x_overlap,
                        "z_overlap": z_overlap,
                        "y_margin": ABOVE_Y_MARGIN,
                    },
                )
                relations.append(rel)
                has_strong_relation = True

            # ---------- near ----------
            if not has_strong_relation:
                if i < j:
                    if is_near(obj_a, obj_b):
                        if should_keep_near(obj_a, obj_b):
                            dist = horizontal_distance(obj_a, obj_b)
                            rel = make_relation_record(
                                subject_obj=obj_a,
                                relation="near",
                                object_obj=obj_b,
                                family="geometric",
                                evidence={
                                    "rule": "horizontal_distance",
                                    "horizontal_distance": dist,
                                    "threshold": NEAR_THRESHOLD,
                                },
                            )
                            relations.append(rel)

                        elif should_keep_near(obj_b, obj_a):
                            dist = horizontal_distance(obj_a, obj_b)
                            rel = make_relation_record(
                                subject_obj=obj_b,
                                relation="near",
                                object_obj=obj_a,
                                family="geometric",
                                evidence={
                                    "rule": "horizontal_distance",
                                    "horizontal_distance": dist,
                                    "threshold": NEAR_THRESHOLD,
                                },
                            )
                            relations.append(rel)

    return relations


def deduplicate_relations(relations):

    # Remove duplicates to avoid identical relation records appearing multiple times.
    seen = set()
    deduped = []

    for rel in relations:
        key = (
            rel["subject_id"],
            rel["relation"],
            rel["object_id"]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)

    return deduped

def filter_relations(relations):

    filtered = []

    for rel in relations:
        verbalizable = should_mark_verbalizable(rel)
        rel["verbalizable"] = verbalizable
        rel["candidate_for_text"] = verbalizable
        filtered.append(rel)

    return filtered

def expand_inverse_relations(relations):

    expanded = list(relations)

    for rel in relations:
        forward_relation = rel["relation"]

        if forward_relation not in INVERSE_RELATION_MAP:
            continue

        inverse_relation = INVERSE_RELATION_MAP[forward_relation]

        inverse_record = {
            "subject_id": rel["object_id"],
            "subject_type": rel["object_type"],
            "relation": inverse_relation,
            "object_id": rel["subject_id"],
            "object_type": rel["subject_type"],
            "relation_family": rel["relation_family"],
            "candidate_for_text": rel.get("candidate_for_text", True),
            "verbalizable": rel.get("verbalizable", True),
            "evidence": {
                "source": "inverse_relation",
                "forward_relation": forward_relation,
                "forward_evidence": rel.get("evidence", {}),
            },
            "is_inverse_relation": True,
            "derived_from_relation": forward_relation,
        }

        expanded.append(inverse_record)

    return expanded

def compress_inverse_relations_randomly(relations):

    # For a pair of inverse relations, randomly keep only one expression.
    grouped = {}
    result = []

    inverse_pairs = {
        frozenset({"in", "contains"}),
        frozenset({"on", "supports"}),
        frozenset({"left_of", "right_of"}),
        frozenset({"above", "below"}),
    }

    for rel in relations:
        subject_id = rel["subject_id"]
        object_id = rel["object_id"]
        relation = rel["relation"]

        matched_pair = None
        for pair in inverse_pairs:
            if relation in pair:
                matched_pair = tuple(sorted(pair))
                break

        if matched_pair is None:
            result.append(rel)
            continue

        obj_pair = tuple(sorted([subject_id, object_id]))
        group_key = (matched_pair, obj_pair)

        grouped.setdefault(group_key, []).append(rel)

    for group in grouped.values():
        result.append(random.choice(group))

    return result

def build_relations(scene_data):
    objects = scene_data["objects"]

    # 1) First construct forward relations
    topo_relations = build_topological_relations(objects)
    geometric_relations = build_geometric_relations(objects)

    relations = topo_relations + geometric_relations
    relations = deduplicate_relations(relations)
    relations = filter_relations(relations)

    # 2) Then derive inverse relations from the forward ones
    relations = expand_inverse_relations(relations)
    relations = deduplicate_relations(relations)

    # 3) Randomly collapse inverse pairs, keeping only one direction
    relations = compress_inverse_relations_randomly(relations)

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
            "inverse_relation_map": INVERSE_RELATION_MAP,
            "relations": relations
        }

        out_filename = filename.replace(".json", "_relations.json")
        out_path = os.path.join(STEP2_DIR, out_filename)

        save_json(out_path, out_data)

        print(f"{scene_data['scene']}: {len(relations)} relations saved.")


if __name__ == "__main__":
    main()
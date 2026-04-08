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

# 几何关系阈值：后续可以继续调
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
    """
    根据 parent 的类型判断是 in 还是 on
    """
    parent_type = parent_obj["objectType"]
    if parent_type in SURFACE_TYPES:
        return "on"
    else:
        return "in"


def get_extent(obj):
    """
    根据 bbox.center 和 bbox.size 计算物体占据的空间区间。
    如果 bbox 缺失，则退回到 position 且 size=0 的退化情况。
    """
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
    """
    计算 x-z 平面的欧氏距离。
    near 第一版先基于它判断。
    """
    p1 = obj1.get("position", {}) or {}
    p2 = obj2.get("position", {}) or {}

    x1 = float(p1.get("x", 0.0))
    z1 = float(p1.get("z", 0.0))
    x2 = float(p2.get("x", 0.0))
    z2 = float(p2.get("z", 0.0))

    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


def is_left_of(obj_a, obj_b, margin=LEFT_MARGIN):
    """
    A 在 B 左边：
    用 x 轴 bbox 区间分离判断。
    """
    ea = get_extent(obj_a)
    eb = get_extent(obj_b)
    return ea["x_max"] < eb["x_min"] - margin


def is_above(obj_a, obj_b, y_margin=ABOVE_Y_MARGIN):
    """
    A 在 B 上方：
    1) A 的底部高于 B 的顶部
    2) 且两者在 x/z 投影上有重叠
    """
    ea = get_extent(obj_a)
    eb = get_extent(obj_b)

    x_overlap = has_overlap_1d(ea["x_min"], ea["x_max"], eb["x_min"], eb["x_max"])
    z_overlap = has_overlap_1d(ea["z_min"], ea["z_max"], eb["z_min"], eb["z_max"])
    y_clear = ea["y_min"] > eb["y_max"] + y_margin

    return x_overlap and z_overlap and y_clear


def is_near(obj_a, obj_b, threshold=NEAR_THRESHOLD):
    """
    A 靠近 B：
    第一版先用简单水平距离阈值。
    """
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
    """
    判断对象是否属于较大型、较结构性的物体。
    用于筛掉一部分机械但不太适合文本表达的几何关系。
    """
    return obj.get("objectType") in STRUCTURAL_TYPES


def should_keep_left_of(obj_a, obj_b):
    """
    left_of 的第一轮筛选规则：
    - 如果双方都是结构对象，则不保留
    - 否则保留
    """
    if is_structural_object(obj_a) and is_structural_object(obj_b):
        return False
    return True


def should_keep_above(obj_a, obj_b):
    """
    above 的第一轮筛选规则：
    - 如果双方 objectType 完全相同且都是结构对象，例如 Drawer above Drawer，
      先过滤掉
    - 如果双方都是结构对象，也先过滤掉
    """
    same_type = obj_a.get("objectType") == obj_b.get("objectType")
    if same_type and is_structural_object(obj_a) and is_structural_object(obj_b):
        return False

    if is_structural_object(obj_a) and is_structural_object(obj_b):
        return False

    return True

def should_keep_near(obj_a, obj_b):
    """
    near 的筛选规则（当前推荐版）：
    1) 不保留 结构对象 -> 结构对象
    2) 不保留 结构对象 -> 小物体
    3) 允许 小物体 -> 结构对象
    4) 允许 小物体 -> 小物体

    这样可以避免：
    - Drawer near Spoon
    - Cabinet near Drawer
    这类虽然几何成立、但不太自然的表达。
    """
    a_structural = is_structural_object(obj_a)
    b_structural = is_structural_object(obj_b)

    # 结构 -> 结构：不保留
    if a_structural and b_structural:
        return False

    # 结构 -> 小物体：不保留
    if a_structural and not b_structural:
        return False

    # 小物体 -> 结构，或 小物体 -> 小物体：保留
    return True

def should_mark_verbalizable(rel):
    """
    给关系打 verbalizable / candidate_for_text 标记。
    这里先做一版保守规则：
    - topological 关系默认可表达
    - near 默认可表达，但它本身已被压缩
    - above / left_of 若涉及双方都是结构对象，则不建议进入文本
    """
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
    """
    基于 parentReceptacles 恢复 in / on
    """
    obj_index = build_object_index(objects)
    relations = []

    for obj in objects:
        for parent_id in obj.get("parentReceptacles", []):
            if parent_id not in obj_index:
                continue  # 只保留 selected objects 内部关系

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
    """
    改进版：
    1) left_of / above 加第一轮筛选
    2) near 不双向保留（只 i < j）
    3) 如果已有强关系（in/on/left_of/above），则不保留 near
    4) near 增加方向性筛选：不允许结构对象做 subject
    5) 给几何关系增加 evidence
    """
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

            # 如果 A -> B 已有拓扑关系，则跳过该方向上的几何关系
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
            # near 只作 fallback，且不双向保留
            if not has_strong_relation:
                if i < j:
                    if is_near(obj_a, obj_b):
                        # 优先尝试 obj_a -> obj_b
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

                        # 如果 obj_a -> obj_b 不合适，但反向合适，则保留反向
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
    """
    去重，避免完全相同的关系记录重复出现。
    """
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
    """
    当前先不删除太多关系，主要做两件事：
    1) 给关系打 verbalizable / candidate_for_text 标记
    2) 为以后更强的过滤逻辑预留统一入口
    """
    filtered = []

    for rel in relations:
        verbalizable = should_mark_verbalizable(rel)
        rel["verbalizable"] = verbalizable
        rel["candidate_for_text"] = verbalizable
        filtered.append(rel)

    return filtered

def expand_inverse_relations(relations):
    """
    根据 INVERSE_RELATION_MAP 为正向关系补充反向关系。

    当前只为以下关系生成反向关系：
    - in -> contains
    - on -> supports
    - left_of -> right_of
    - above -> below

    near 当前不生成反向关系，因为它已被视为弱关系/补充关系。

    反向关系保留：
    - relation_family
    - candidate_for_text
    - verbalizable
    - evidence（附加 derived_from 说明）
    并新增：
    - is_inverse_relation
    - derived_from_relation
    """
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
    """
    对互为反向的一组关系，只随机保留其中一个表达。

    例如：
    - A left_of B
    - B right_of A

    只保留其中一条，且不是固定某个方向，而是随机选择。

    适用关系：
    - in / contains
    - on / supports
    - left_of / right_of
    - above / below

    near 不参与，因为目前本来就不扩反向。
    """
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

        # 不属于互逆压缩集合的关系，直接保留
        if matched_pair is None:
            result.append(rel)
            continue

        # 用无方向 object pair 分组
        obj_pair = tuple(sorted([subject_id, object_id]))
        group_key = (matched_pair, obj_pair)

        grouped.setdefault(group_key, []).append(rel)

    for group in grouped.values():
        # 随机保留该组中的一条
        result.append(random.choice(group))

    return result

def build_relations(scene_data):
    objects = scene_data["objects"]

    # 1. 先构造正向关系
    topo_relations = build_topological_relations(objects)
    geometric_relations = build_geometric_relations(objects)

    relations = topo_relations + geometric_relations
    relations = deduplicate_relations(relations)
    relations = filter_relations(relations)

    # 2. 再基于正向关系扩展反向关系
    relations = expand_inverse_relations(relations)
    relations = deduplicate_relations(relations)

    # 3. 对互逆关系随机压缩，只保留一个方向
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
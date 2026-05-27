# ============================================================
# pilot_4/config.py
# Central configuration for Pilot 4 pipeline
# ============================================================

import os

# ============================================================
# Project paths
# ============================================================

PILOT_ROOT = "/content/pilot_4"
DATA_DIR = os.path.join(PILOT_ROOT, "data")
SCRIPTS_DIR = os.path.join(PILOT_ROOT, "scripts")

STEP1_DIR = os.path.join(DATA_DIR, "step1_chunk_ground_truth")
STEP2_DIR = os.path.join(DATA_DIR, "step2_chunk_relations")

STEP3A_CANONICAL_DIR = os.path.join(DATA_DIR, "step3A_canonical_text_preserve_text_direction")
STEP3A_DIVERSE_DIR = os.path.join(DATA_DIR, "step3A_diverse_text_preserve_text_direction")

STEP4_INPUT_DIR = STEP3A_DIVERSE_DIR
STEP4_OUTPUT_DIR = os.path.join(DATA_DIR, "step4_probe_datasets_preserve_text_direction")

# ============================================================
# Global reproducibility
# ============================================================

RANDOM_SEED = 42

# ============================================================
# Scenes
# ============================================================

SCENE_START = 1
SCENE_END = 6
SCENES = [f"FloorPlan{i}" for i in range(SCENE_START, SCENE_END + 1)]

# ============================================================
# AI2-THOR controller
# ============================================================

CONTROLLER_WIDTH = 300
CONTROLLER_HEIGHT = 300
AGENT_MODE = "default"

# ============================================================
# Step 1: object filtering and chunk / cluster construction
# ============================================================

USE_PICKUPABLE_ONLY = False

EXCLUDED_TYPES = {
    "Floor",
    "Wall",
    "Ceiling",
}

ANCHOR_TYPES = {
    "CounterTop",
    "TableTop",
    "Desk",
    "Shelf",
    "Sink",
    "StoveBurner",
    "DiningTable",
    "SideTable",
    "Cabinet",
    "Drawer",
    "Fridge",
    "Microwave",
    "Safe",
}

STRUCTURAL_TYPES = {
    "Cabinet",
    "CoffeeMachine",
    "CounterTop",
    "Desk",
    "DiningTable",
    "Drawer",
    "Fridge",
    "Microwave",
    "Safe",
    "Shelf",
    "SideTable",
    "Sink",
    "StoveBurner",
    "TVStand",
    "TableTop",
    "Toaster",
}

GRID_NX = 3
GRID_NZ = 3

PARENT_COMPLETION_RADIUS = 1.5
CHILD_COMPLETION_RADIUS = 1.5
MAX_CHILDREN_PER_ANCHOR = 5

NEAR_GRAPH_THRESHOLD = 0.8

MIN_OBJECTS_PER_CLUSTER = 4
MAX_OBJECTS_PER_CLUSTER = 15
MIN_ANCHORS_PER_CLUSTER = 1
MIN_SMALL_OBJECTS_PER_CLUSTER = 1
MAX_CLUSTERS_PER_CHUNK = 2
KEEP_WEAK_CLUSTERS = False

# ============================================================
# Shared relation labels
# ============================================================

NONE_RELATION_LABEL = "none"

INVERSE_RELATION_MAP = {
    "in": "contains",
    "contains": "in",
    "on": "supports",
    "supports": "on",
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "near": "near",
    "none": "none",
}

INVERSE_RELATION_GROUPS = {
    frozenset({"in", "contains"}),
    frozenset({"on", "supports"}),
    frozenset({"left_of", "right_of"}),
    frozenset({"above", "below"}),
}

DIRECTIONAL_RELATIONS = {
    "in",
    "contains",
    "on",
    "supports",
    "left_of",
    "right_of",
    "above",
    "below",
}

SYMMETRIC_OR_WEAK_RELATIONS = {
    "near",
}

# ============================================================
# Step 2: relation extraction
# ============================================================

SURFACE_TYPES = {
    "CounterTop",
    "TableTop",
    "Desk",
    "Shelf",
    "Sink",
    "StoveBurner",
    "DiningTable",
    "SideTable",
}

CONTAINER_LIKE_TYPES = {
    "Cabinet",
    "Drawer",
    "Fridge",
    "Microwave",
    "Safe",
    "Cup",
    "Mug",
    "Bowl",
    "Plate",
    "Pan",
    "Pot",
}

LEFT_MARGIN = 0.05
ABOVE_Y_MARGIN = 0.03
NEAR_THRESHOLD = 0.50

LEFT_REQUIRE_Z_OVERLAP_OR_NEAR = True
LEFT_MAX_Z_CENTER_DISTANCE = 0.80

MAX_NEAR_RELATIONS_PER_CLUSTER = 12
MAX_TEXT_CANDIDATE_RELATIONS_PER_CLUSTER = 12

BUILD_GEOMETRY_PAIR_RECORDS = True
MAX_GEOMETRY_PAIRS_PER_CLUSTER = 80

# ============================================================
# Step 3A: paragraph generation
# ============================================================

EXPERIMENT_NAME = "A_relation_classification_preserve_text_direction"

MAX_PARAGRAPHS_PER_CLUSTER = 4

MIN_RELATIONS_PER_PARAGRAPH = 6
TARGET_RELATIONS_PER_PARAGRAPH = 16
MAX_RELATIONS_PER_PARAGRAPH = 18

USE_PREFERRED_RELATIONS_FIRST = True

ALLOW_NEAR_IN_TEXT = True
MAX_NEAR_SENTENCES_PER_PARAGRAPH = 2

ALLOW_INVERSE_TOPOLOGICAL_IN_TEXT = True

INCLUDE_NONE_RELATION_PAIRS = True

TARGET_OBJECT_COVERAGE_RATIO = 0.75
MAX_OBJECTS_PER_PARAGRAPH = 12

NEW_OBJECT_BONUS = 60
TWO_NEW_OBJECT_BONUS = 30

ALLOW_REDUNDANT_RELATIONS = True
MAX_REDUNDANT_RELATIONS = 8

RELATION_SELECTION_JITTER = 35
REUSED_RELATION_PENALTY = 80

CANONICAL_MODE = "canonical"
DIVERSE_MODE = "diverse"
RUN_MODE = DIVERSE_MODE

INTRO_TEMPLATES = [
    "In this local kitchen area,",
    "In this part of the kitchen,",
    "Within this local scene,",
    "In the described kitchen area,",
]

RELATION_TEMPLATES = {
    "in": [
        "{subj} is in {obj}.",
        "{subj} is inside {obj}.",
        "{subj} is contained in {obj}.",
        "{subj} is kept in {obj}.",
        "{subj} sits inside {obj}.",
        "{subj} can be found in {obj}.",
        "{subj} is placed inside {obj}.",
        "You can find {subj} in {obj}.",
        "Inside {obj}, there is {subj}.",
        "Inside {obj}, you can see {subj}.",
    ],
    "on": [
        "{subj} is on {obj}.",
        "{subj} is on top of {obj}.",
        "{subj} is resting on {obj}.",
        "{subj} sits on {obj}.",
        "{subj} has been placed on {obj}.",
        "{subj} lies on {obj}.",
        "You can see {subj} on {obj}.",
        "On {obj}, there is {subj}.",
        "On top of {obj}, there is {subj}.",
    ],
    "left_of": [
        "{subj} is to the left of {obj}.",
        "{subj} is on the left side of {obj}.",
        "{subj} sits to the left of {obj}.",
        "{subj} is positioned to the left of {obj}.",
        "{subj} can be seen to the left of {obj}.",
        "To the left of {obj}, there is {subj}.",
    ],
    "right_of": [
        "{subj} is to the right of {obj}.",
        "{subj} is on the right side of {obj}.",
        "{subj} sits to the right of {obj}.",
        "{subj} is positioned to the right of {obj}.",
        "{subj} can be seen to the right of {obj}.",
        "To the right of {obj}, there is {subj}.",
    ],
    "above": [
        "{subj} is above {obj}.",
        "{subj} is positioned above {obj}.",
        "{subj} is located above {obj}.",
        "{subj} sits above {obj}.",
        "{subj} appears above {obj}.",
        "Above {obj}, there is {subj}.",
        "Directly above {obj}, you can see {subj}.",
    ],
    "below": [
        "{subj} is below {obj}.",
        "{subj} is underneath {obj}.",
        "{subj} is located below {obj}.",
        "{subj} sits below {obj}.",
        "{subj} appears below {obj}.",
        "Below {obj}, there is {subj}.",
        "Directly below {obj}, you can see {subj}.",
    ],
    "near": [
        "{subj} is near {obj}.",
        "{subj} is close to {obj}.",
        "{subj} is nearby {obj}.",
        "{subj} is not far from {obj}.",
        "{subj} is positioned near {obj}.",
        "{subj} is located close to {obj}.",
        "{subj} can be found near {obj}.",
        "Near {obj}, there is {subj}.",
    ],
    "supports": [
        "{subj} supports {obj}.",
        "{subj} is holding up {obj}.",
        "{subj} has {obj} on top of it.",
        "{obj} is resting on {subj}.",
        "On {subj}, there is {obj}.",
    ],
    "contains": [
        "{subj} contains {obj}.",
        "{subj} has {obj} inside it.",
        "{subj} holds {obj}.",
        "{subj} includes {obj}.",
        "Inside {subj}, there is {obj}.",
        "{obj} is inside {subj}.",
    ],
}

CANONICAL_TEMPLATE = {
    relation: templates[0]
    for relation, templates in RELATION_TEMPLATES.items()
}

TEXT_RELATION_PRIORITY = {
    "on": 100,
    "in": 100,
    "above": 90,
    "below": 90,
    "left_of": 80,
    "right_of": 80,
    "near": 50,
    "supports": 40,
    "contains": 40,
}

INVERSE_TO_NATURAL = {
    "supports": "on",
    "contains": "in",
    "right_of": "left_of",
    "below": "above",
}

NATURAL_INVERSE_SWAP_RELATIONS = {
    "supports",
    "contains",
    "right_of",
    "below",
}

# This alias is kept for Step 3 code that uses INVERSE_LABEL_MAP.
INVERSE_LABEL_MAP = INVERSE_RELATION_MAP

# ============================================================
# Step 4: probe dataset construction
# ============================================================

EXPLICIT_DIRECT = "explicit_direct"
EXPLICIT_INVERSE = "explicit_inverse_or_same_sentence_pair"
IMPLICIT_LABELED = "implicit_labeled"
IMPLICIT_NONE = "implicit_none"

MAIN_PROBE_EVIDENCE_TYPES = {
    EXPLICIT_DIRECT,
    EXPLICIT_INVERSE,
}

# ============================================================
# Step 5: hidden-state extraction
# ============================================================

STEP5_MODEL_NAME = "Qwen/Qwen2.5-3B"
STEP5_MODEL_TAG = "Qwen2_5_3B"

STEP5_MAX_LENGTH = 768

STEP5_SOURCE_TYPE = "pilot4_step4_probe_samples"
STEP5_INPUT_TEXT_MODE = "paragraph_probe_samples"
STEP5_FEATURE_ANCHOR = "probe_pair_subject_object"

STEP5_INPUT_DIR = STEP4_OUTPUT_DIR
STEP5_OUTPUT_DIR = os.path.join(DATA_DIR, "step5_hidden_states_qwen2_5_3b_preserve_text_direction")


# ============================================================
# Step 6: Setting B3 scene-split evaluation
# Train direct, test direct
# ============================================================

STEP6_EXPERIMENT_NAME = "settingB3_scene_split_train_direct_test_direct"

STEP6_INPUT_DIR = os.path.join(DATA_DIR, "step5_hidden_states_input")

STEP6_OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "step6_settingB3_scene_split_train_direct_test_direct_outputs",
)

STEP6_FEATURE_KEY = "layer_diff_features"
STEP6_LABEL_FIELD = "relation"

STEP6_TRAIN_SCENES = [
    "FloorPlan1",
    "FloorPlan2",
    "FloorPlan3",
    "FloorPlan4",
]

STEP6_TEST_SCENES = [
    "FloorPlan5",
    "FloorPlan6",
]

STEP6_REQUIRE_EXPLICIT_SCENE_SPLIT = True

STEP6_DIRECTION_FILTER_PROTOCOL = "same_direction"

STEP6_TRAIN_DIRECTION_SELECTION_MODE = "direct"
STEP6_TEST_DIRECTION_SELECTION_MODE = "direct"

STEP6_FILTER_TRAIN_BY_DIRECTION = True
STEP6_FILTER_TEST_BY_DIRECTION = True

STEP6_LOGREG_MAX_ITER = 5000
STEP6_LOGREG_C = 1.0

STEP6_PRINT_DATASET_SUMMARY = True
STEP6_PRINT_LAYER_PROGRESS = True
STEP6_PRINT_TOP_LAYERS = True
STEP6_NUM_TOP_LAYERS_TO_PRINT = 10


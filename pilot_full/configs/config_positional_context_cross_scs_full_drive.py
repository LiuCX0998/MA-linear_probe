# Central configuration for full pipeline

import os

# Project paths

PILOT_ROOT = "/content/pilot_full"
SCRIPTS_DIR = os.path.join(PILOT_ROOT, "scripts")

DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/linear_probe_full"
PIPELINE_ROOT = os.path.join(DRIVE_ROOT, "data_prepare_step1_3")
PIPELINE_DATA_DIR = os.path.join(PIPELINE_ROOT, "data")

DATA_DIR = PIPELINE_DATA_DIR

# Pipeline data directories

STEP1_OUTPUT_DIR = os.path.join(PIPELINE_DATA_DIR, "step1_gt")

STEP2_INPUT_DIR = STEP1_OUTPUT_DIR
STEP2_OUTPUT_DIR = os.path.join(PIPELINE_DATA_DIR, "step2_rel")

STEP3_INPUT_DIR = STEP2_OUTPUT_DIR
STEP3_CANONICAL_OUTPUT_DIR = os.path.join(PIPELINE_DATA_DIR, "step3_can")
STEP3_DIVERSE_OUTPUT_DIR = os.path.join(PIPELINE_DATA_DIR, "step3_div")

STEP4_INPUT_DIR = STEP3_DIVERSE_OUTPUT_DIR
STEP4_OUTPUT_DIR = os.path.join(PIPELINE_DATA_DIR, "step4_probe")


# Experiment naming

PIPELINE_TAG = "pilot_full"
DATA_TAG = "ithor120"
TEXT_DIRECTION_TAG = "ptd"  # preserve_text_direction

def make_model_tag(model_name: str) -> str:
    """
        Qwen/Qwen2.5-3B        -> qwen2_5_3b
        Qwen/Qwen2.5-7B        -> qwen2_5_7b
    """
    tag = model_name.split("/")[-1]
    tag = tag.replace(".", "_")
    tag = tag.replace("-", "_")
    tag = tag.replace(" ", "_")
    tag = tag.lower()
    return tag

# Global reproducibility

RANDOM_SEED = 42


# Scenes

# Full iTHOR scene set:
#   Kitchens:      FloorPlan1   - FloorPlan30
#   Living rooms:  FloorPlan201 - FloorPlan230
#   Bedrooms:      FloorPlan301 - FloorPlan330
#   Bathrooms:     FloorPlan401 - FloorPlan430

KITCHEN_SCENES = [f"FloorPlan{i}" for i in range(1, 31)]
LIVING_ROOM_SCENES = [f"FloorPlan{200 + i}" for i in range(1, 31)]
BEDROOM_SCENES = [f"FloorPlan{300 + i}" for i in range(1, 31)]
BATHROOM_SCENES = [f"FloorPlan{400 + i}" for i in range(1, 31)]

SCENES = (
    KITCHEN_SCENES
    + LIVING_ROOM_SCENES
    + BEDROOM_SCENES
    + BATHROOM_SCENES
)

def infer_scene_type_from_floorplan(scene: str) -> str:
    if scene in KITCHEN_SCENES:
        return "kitchen"
    if scene in LIVING_ROOM_SCENES:
        return "living_room"
    if scene in BEDROOM_SCENES:
        return "bedroom"
    if scene in BATHROOM_SCENES:
        return "bathroom"
    return "unknown"

assert len(SCENES) == 120, f"Expected 120 iTHOR scenes, got {len(SCENES)}"

# AI2-THOR controller

CONTROLLER_WIDTH = 300
CONTROLLER_HEIGHT = 300
AGENT_MODE = "default"

# Step 1: object filtering and chunk / cluster construction

USE_PICKUPABLE_ONLY = False

EXCLUDED_TYPES = {
    "Floor",
    "Wall",
    "Ceiling",
}

ANCHOR_TYPES = {
    "ArmChair",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Box",
    "Cabinet",
    "Chair",
    "CoffeeTable",
    "CounterTop",
    "Desk",
    "DiningTable",
    "Drawer",
    "Fridge",
    "GarbageCan",
    "HandTowelHolder",
    "Microwave",
    "Shelf",
    "SideTable",
    "Sink",
    "SinkBasin",
    "Sofa",
    "StoveBurner",
    "Toilet",
    "ToiletPaperHanger",
    "TowelHolder",
}

STRUCTURAL_TYPES = {
    "ArmChair",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Blinds",
    "Cabinet",
    "Chair",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "Desk",
    "DeskLamp",
    "DiningTable",
    "Drawer",
    "Faucet",
    "FloorLamp",
    "Fridge",
    "GarbageCan",
    "HandTowelHolder",
    "HousePlant",
    "LightSwitch",
    "Microwave",
    "Mirror",
    "Painting",
    "Shelf",
    "ShowerDoor",
    "ShowerHead",
    "SideTable",
    "Sink",
    "SinkBasin",
    "Sofa",
    "StoveBurner",
    "StoveKnob",
    "Television",
    "Toaster",
    "Toilet",
    "ToiletPaperHanger",
    "TowelHolder",
    "Window",
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


# Shared relation labels

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


# Step 2: relation extraction

SURFACE_TYPES = {
    "ArmChair",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Chair",
    "CoffeeTable",
    "CounterTop",
    "Desk",
    "DiningTable",
    "HandTowelHolder",
    "Shelf",
    "SideTable",
    "Sink",
    "SinkBasin",
    "Sofa",
    "StoveBurner",
    "Toilet",
    "ToiletPaperHanger",
    "TowelHolder",
}

CONTAINER_LIKE_TYPES = {
    "Bathtub",
    "BathtubBasin",
    "Bowl",
    "Box",
    "Cabinet",
    "Cup",
    "Drawer",
    "Fridge",
    "GarbageCan",
    "Microwave",
    "Mug",
    "Pan",
    "Pot",
    "Sink",
    "SinkBasin",
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


# Step 3: paragraph generation

EXPERIMENT_NAME = "step3_relation_classification_preserve_text_direction"
EXPERIMENT_TAG = "step3_ptd"

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

INTRO_TEMPLATES_BY_SCENE_TYPE = {
    "kitchen": [
        "In this local kitchen area,",
        "In this part of the kitchen,",
        "Within this kitchen scene,",
        "In the described kitchen area,",
    ],
    "living_room": [
        "In this local living room area,",
        "In this part of the living room,",
        "Within this living room scene,",
        "In the described living room area,",
    ],
    "bedroom": [
        "In this local bedroom area,",
        "In this part of the bedroom,",
        "Within this bedroom scene,",
        "In the described bedroom area,",
    ],
    "bathroom": [
        "In this local bathroom area,",
        "In this part of the bathroom,",
        "Within this bathroom scene,",
        "In the described bathroom area,",
    ],
    "unknown": [
        "In this local scene area,",
        "In this part of the scene,",
        "Within this local scene,",
        "In the described scene area,",
    ],
}

INTRO_TEMPLATES = INTRO_TEMPLATES_BY_SCENE_TYPE["unknown"]

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

INVERSE_LABEL_MAP = INVERSE_RELATION_MAP


# Step 4: probe dataset construction

EXPLICIT_DIRECT = "explicit_direct"
EXPLICIT_INVERSE = "explicit_inverse_or_same_sentence_pair"
IMPLICIT_LABELED = "implicit_labeled"
IMPLICIT_NONE = "implicit_none"
IMPLICIT_TRANSITIVE = "implicit_transitive"

MAIN_PROBE_EVIDENCE_TYPES = {
    EXPLICIT_DIRECT,
    EXPLICIT_INVERSE,
}

# Step4 reads Step3 scene-level JSON files directly.
STEP4_INPUT_SOURCE = "scene_json"

STEP4_INPUT_DIR = (
    STEP3_DIVERSE_OUTPUT_DIR
    if RUN_MODE == DIVERSE_MODE
    else STEP3_CANONICAL_OUTPUT_DIR
)

STEP4_SCENE_JSON_SUFFIX = f"_step3_text_{RUN_MODE}.json"

STEP4_OUTPUT_DIR = os.path.join(PIPELINE_DATA_DIR, "step4_probe")


# Step4 pair-level probe output

STEP4_OUTPUT_FILE = os.path.join(
    STEP4_OUTPUT_DIR,
    f"step4_probe_samples_{RUN_MODE}_all.jsonl",
)

STEP4_MANIFEST_FILE = os.path.join(
    STEP4_OUTPUT_DIR,
    f"step4_manifest_{RUN_MODE}.json",
)

STEP4_PREVIEW_CSV_FILE = os.path.join(
    STEP4_OUTPUT_DIR,
    f"step4_probe_samples_{RUN_MODE}_preview.csv",
)

# Keep all pair types at Step4.
# Step6 can later filter explicit-only / labeled-only / non-none as needed.
STEP4_KEEP_EVIDENCE_TYPES = {
    EXPLICIT_DIRECT,
    EXPLICIT_INVERSE,
    IMPLICIT_LABELED,
    IMPLICIT_NONE,
}

STEP4_INCLUDE_NONE_LABEL = True

STEP4_WRITE_PER_SCENE_FILES = True


# Step4 composition / implicit-transitive output

STEP4_COMPOSITION_OUTPUT_FILE = os.path.join(
    STEP4_OUTPUT_DIR,
    f"step4_composition_samples_{RUN_MODE}_all.jsonl",
)

STEP4_COMPOSITION_MANIFEST_FILE = os.path.join(
    STEP4_OUTPUT_DIR,
    f"step4_composition_manifest_{RUN_MODE}.json",
)

STEP4_COMPOSITION_PREVIEW_CSV_FILE = os.path.join(
    STEP4_OUTPUT_DIR,
    f"step4_composition_samples_{RUN_MODE}_preview.csv",
)

STEP4_WRITE_COMPOSITION_PER_SCENE_FILES = True

# Only use relations that are plausibly transitive under the current geometric relation scheme.
STEP4_COMPOSITION_TARGET_RELATIONS = {
    "left_of",
    "right_of",
    "above",
    "below",
}

# The target A-C relation must be an implicit_labeled pair from Step3 pair_targets.
STEP4_COMPOSITION_REQUIRE_TARGET_IMPLICIT_LABELED = True

# Exclude cases where A-C or C-A is explicitly written in the paragraph.
STEP4_COMPOSITION_EXCLUDE_DIRECT_OR_INVERSE_EXPLICIT_TARGET = True

# Stronger leakage control:
# if A and C occur as an explicit pair with any relation, exclude the sample.
STEP4_COMPOSITION_EXCLUDE_ANY_EXPLICIT_TARGET_PAIR = True

# Do not require a unique support path.
# If A-C can be supported by A-B-C and A-D-C, keep one A-C sample and record both paths.
STEP4_COMPOSITION_REQUIRE_UNIQUE_SUPPORT_PATH = False
STEP4_COMPOSITION_KEEP_ALL_SUPPORT_PATHS = True

# Which surface-form support rules are allowed.
# chain_same_direction:
#   A R B + B R C -> A R C
# shared_anchor_opposite_sides:
#   A R B + C inverse(R) B -> A R C
# anchor_between_reversed_surface_form:
#   B inverse(R) A + B R C -> A R C
STEP4_COMPOSITION_ALLOWED_RULES = {
    "chain_same_direction",
    "shared_anchor_opposite_sides",
    "anchor_between_reversed_surface_form",
}


# Step 5: hidden-state extraction

# Available models.
STEP5_AVAILABLE_MODEL_NAMES = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-3B",
]

# Active model for Step5.
STEP5_MODEL_NAME = "Qwen/Qwen2.5-3B"
STEP5_MODEL_TAG = make_model_tag(STEP5_MODEL_NAME)

STEP5_MAX_LENGTH = 768

STEP5_INPUT_DIR = STEP4_OUTPUT_DIR

# Step4 sample families to process:
#   pair        -> ordinary pair-level probe samples
#   composition -> implicit_transitive samples
STEP5_SAMPLE_FAMILIES = [
    "pair",
    "composition",
]

# Prefer per-scene Step4 files to produce one .pt per scene.
STEP5_USE_PER_SCENE_FILES = True

STEP5_PAIR_INPUT_GLOB = f"FloorPlan*_step4_probe_samples_{RUN_MODE}.jsonl"
STEP5_COMPOSITION_INPUT_GLOB = f"FloorPlan*_step4_composition_samples_{RUN_MODE}.jsonl"

STEP5_PAIR_ALL_INPUT_FILE = STEP4_OUTPUT_FILE
STEP5_COMPOSITION_ALL_INPUT_FILE = STEP4_COMPOSITION_OUTPUT_FILE

STEP5_INPUT_TEXT_FIELD = "text"
STEP5_SUBJECT_FIELD = "subject_uid"
STEP5_OBJECT_FIELD = "object_uid"

STEP5_SOURCE_TYPE = f"{PIPELINE_TAG}_step4_samples"
STEP5_INPUT_TEXT_MODE = "paragraph_text"
STEP5_FEATURE_ANCHOR = "subject_object_alias_spans"

# Only row["text"] is sent to the LLM.
# Labels, evidence types, geometry labels, and supporting paths are copied as metadata only.
STEP5_FORWARD_TEXT_ONLY = True

# subject/object representation = mean over all visible occurrences of that alias.
STEP5_USE_ALL_ALIAS_OCCURRENCES = True

STEP5_SAVE_LAYER_SUBJECT_OBJECT = True
STEP5_SAVE_LAYER_DIFF = True
STEP5_SAVE_LAYER_CONCAT = True

# Store features in float16 to reduce disk usage.
STEP5_FEATURE_DTYPE = "float16"

STEP5_OUTPUT_ROOT = PIPELINE_DATA_DIR

# Output naming
STEP5_OUTPUT_DIR_TEMPLATE = "step5_hs_{sample_family}_{model_tag}"
STEP5_OUTPUT_PT_FILENAME_TEMPLATE = "{stem}_step5_hs_{model_tag}.pt"
STEP5_RUN_SUMMARY_FILENAME_TEMPLATE = "step5_hidden_state_export_summary_{sample_family}_{model_tag}.json"
STEP5_GLOBAL_SUMMARY_FILENAME_TEMPLATE = "step5_hidden_state_global_summary_{model_tag}_{run_mode}.json"


def make_step5_output_dir_for_model(model_name: str, sample_family: str) -> str:
    model_tag = make_model_tag(model_name)

    dirname = STEP5_OUTPUT_DIR_TEMPLATE.format(
        sample_family=sample_family,
        model_tag=model_tag,
        run_mode=RUN_MODE,
    )

    return os.path.join(
        STEP5_OUTPUT_ROOT,
        dirname,
    )


def make_step5_output_dir(sample_family: str) -> str:
    return make_step5_output_dir_for_model(
        STEP5_MODEL_NAME,
        sample_family,
    )


def make_step5_output_pt_filename(source_filename: str, sample_family: str) -> str:
    stem = source_filename.replace(".jsonl", "")

    return STEP5_OUTPUT_PT_FILENAME_TEMPLATE.format(
        stem=stem,
        sample_family=sample_family,
        model_tag=STEP5_MODEL_TAG,
        run_mode=RUN_MODE,
    )


def make_step5_run_summary_filename(sample_family: str) -> str:
    return STEP5_RUN_SUMMARY_FILENAME_TEMPLATE.format(
        sample_family=sample_family,
        model_tag=STEP5_MODEL_TAG,
        run_mode=RUN_MODE,
    )


def make_step5_global_summary_path() -> str:
    filename = STEP5_GLOBAL_SUMMARY_FILENAME_TEMPLATE.format(
        model_tag=STEP5_MODEL_TAG,
        run_mode=RUN_MODE,
    )

    return os.path.join(
        PIPELINE_DATA_DIR,
        filename,
    )

# Step 6: configurable layer-wise linear probing

# Step 6 global identity
STEP6_STAGE_TAG = "step6"
STEP6_TASK_TAG = "probe"
STEP6_PROBE_TYPE = "layerwise_logreg"


# Step 6 current experiment control

STEP6_SPLIT_TAG = "heldout_bathroom_test_seen_unseen_dtp_only"
STEP6_LABEL_SUBSET_MODE = "all_directional"

# Test-time ablation mode.
# This does not change the training set.
# It only partitions the held-out bathroom test set.
STEP6_TEST_ABLATION_MODE = "seen_unseen_train_directed_type_pair"

# Main pair-coverage key used for seen/unseen partitioning.
STEP6_TEST_PAIR_COVERAGE_KEY = "directed_type_pair"

# Keep the full bathroom test subset as a reference in addition to seen/unseen.
STEP6_TEST_INCLUDE_FULL_BATHROOM = False

STEP6_TEST_SUBSET_OVERALL_NAME = "overall"
STEP6_TEST_SUBSET_SEEN_NAME = "seen_train_directed_type_pair"
STEP6_TEST_SUBSET_UNSEEN_NAME = "unseen_train_directed_type_pair"

# Current ablation run evaluates only seen and unseen test subsets.
STEP6_TEST_SUBSETS_TO_EVALUATE = [
    STEP6_TEST_SUBSET_SEEN_NAME,
    STEP6_TEST_SUBSET_UNSEEN_NAME,
]

# Active model for Step6 matches the Step5 model.
STEP6_MODEL_NAME = STEP5_MODEL_NAME
STEP6_MODEL_TAG = make_model_tag(STEP6_MODEL_NAME)


# Step 6 input source family
# Step5 can produce at least two hidden-state families:
#   pair        -> ordinary pair-level samples
#   composition -> implicit-transitive / composition samples

STEP6_SAMPLE_FAMILIES = ["pair"]
STEP6_SAMPLE_FAMILY = "pair"

STEP6_AVAILABLE_SAMPLE_FAMILIES = {
    "pair": {
        "description": "ordinary_pair_level_step5_hidden_states",
        "step5_output_dir": make_step5_output_dir_for_model(
            STEP6_MODEL_NAME,
            "pair",
        ),
    },
    "composition": {
        "description": "composition_or_implicit_transitive_step5_hidden_states",
        "step5_output_dir": make_step5_output_dir_for_model(
            STEP6_MODEL_NAME,
            "composition",
        ),
    },
}


def make_step6_input_dir_for_family(
    model_name: str,
    sample_family: str,
) -> str:
    """
    Return the Google Drive Step5 hidden-state directory for a Step6 input family.
    This reuses Step5 naming instead of hard-coding paths.
    """
    return make_step5_output_dir_for_model(
        model_name=model_name,
        sample_family=sample_family,
    )


def make_step6_input_dirs(
    model_name: str,
    sample_families,
) -> dict:
    """
    Return a dict:
        sample_family -> Step5 output dir
    """
    return {
        sample_family: make_step6_input_dir_for_family(
            model_name=model_name,
            sample_family=sample_family,
        )
        for sample_family in sample_families
    }


STEP6_USE_STEP5_ZIP_INPUT = False

STEP5_OUTPUT_ZIP_FILENAME_TEMPLATE = "step5_hs_{sample_family}_{model_tag}.zip"


def make_step5_output_zip_path_for_model(
    model_name: str,
    sample_family: str,
) -> str:
    model_tag = make_model_tag(model_name)

    filename = STEP5_OUTPUT_ZIP_FILENAME_TEMPLATE.format(
        sample_family=sample_family,
        model_tag=model_tag,
        run_mode=RUN_MODE,
    )

    return os.path.join(
        PIPELINE_DATA_DIR,
        filename,
    )


def make_step6_input_zip_paths(
    model_name: str,
    sample_families,
) -> dict:
    return {
        sample_family: make_step5_output_zip_path_for_model(
            model_name=model_name,
            sample_family=sample_family,
        )
        for sample_family in sample_families
    }


# Step 6 current experiment

STEP6_EXPERIMENT_ID = "pair_explicit_direct_relation_ld_test_seen_unseen_dtp_only"
STEP6_EXPERIMENT_SHORT_NAME = "pair_ed_rel_ld_seen_unseen_dtp_only"
STEP6_EXPERIMENT_DESCRIPTION = (
    "heldout_bathroom_cross_scene_layerwise_logreg_for_explicit_direct_pair_relation_labels_"
    "evaluated_only_on_seen_and_unseen_directed_type_pair_test_subsets"
)
STEP6_SCRIPT_FAMILY = "pair_relation_layerwise_logreg"

STEP6_FEATURE_KEY = "layer_diff_features"
STEP6_LABEL_FIELD = "relation"

STEP6_KEEP_EVIDENCE_TYPES = {
    EXPLICIT_DIRECT,
}

STEP6_DROP_NONE_LABEL = True
STEP6_DROP_EMPTY_LABEL = True

STEP6_TARGET_TAG = "relation"
STEP6_EVIDENCE_TAG = "explicit_direct"
STEP6_FEATURE_TAG = "ld"
STEP6_LABEL_SPACE_TAG = "non_none_relation"


# Inherited naming
# Naming rule:
#   step6_<pipeline>_<data>_<run_mode>_<model>_<sample_family>_<evidence>_<target>_<feature>_<split>

STEP6_INPUT_FAMILY_TAG = STEP6_SAMPLE_FAMILY

STEP6_EXPERIMENT_NAME = "_".join([
    STEP6_STAGE_TAG,
    PIPELINE_TAG,
    DATA_TAG,
    RUN_MODE,
    STEP6_MODEL_TAG,
    STEP6_INPUT_FAMILY_TAG,
    STEP6_EVIDENCE_TAG,
    STEP6_TARGET_TAG,
    STEP6_FEATURE_TAG,
    STEP6_SPLIT_TAG,
])

# Step 6 input/output directories

STEP6_INPUT_DIR = make_step6_input_dir_for_family(
    model_name=STEP6_MODEL_NAME,
    sample_family=STEP6_SAMPLE_FAMILY,
)

STEP6_INPUT_DIRS = make_step6_input_dirs(
    model_name=STEP6_MODEL_NAME,
    sample_families=STEP6_SAMPLE_FAMILIES,
)

STEP6_INPUT_ZIP_PATHS = make_step6_input_zip_paths(
    model_name=STEP6_MODEL_NAME,
    sample_families=STEP6_SAMPLE_FAMILIES,
)

STEP6_OUTPUT_ROOT = os.path.join(
    PIPELINE_DATA_DIR,
    "step6_probe",
)

STEP6_OUTPUT_DIR = os.path.join(
    STEP6_OUTPUT_ROOT,
    STEP6_EXPERIMENT_NAME,
)

STEP6_RESULT_JSON = os.path.join(
    STEP6_OUTPUT_DIR,
    f"{STEP6_EXPERIMENT_NAME}_full_results.json",
)

STEP6_LAYER_SCORES_CSV = os.path.join(
    STEP6_OUTPUT_DIR,
    f"{STEP6_EXPERIMENT_NAME}_layer_scores.csv",
)

STEP6_LABEL_METRICS_CSV = os.path.join(
    STEP6_OUTPUT_DIR,
    f"{STEP6_EXPERIMENT_NAME}_per_layer_label_metrics.csv",
)

STEP6_RECALL_MATRIX_LONG_CSV = os.path.join(
    STEP6_OUTPUT_DIR,
    f"{STEP6_EXPERIMENT_NAME}_recall_matrix_long.csv",
)

STEP6_TEST_PREDICTIONS_CSV = os.path.join(
    STEP6_OUTPUT_DIR,
    f"{STEP6_EXPERIMENT_NAME}_test_predictions_by_layer.csv",
)

STEP6_MANIFEST_JSON = os.path.join(
    STEP6_OUTPUT_DIR,
    f"{STEP6_EXPERIMENT_NAME}_manifest.json",
)

# Scene split setting
# Current cross-scene split:
#   Train: kitchen + living_room + bedroom
#   Test:  bathroom
#
# Held-out bathroom scene-type split.

STEP6_TRAIN_SCENES = (
    KITCHEN_SCENES
    + LIVING_ROOM_SCENES
    + BEDROOM_SCENES
)

STEP6_TEST_SCENES = BATHROOM_SCENES

assert not (set(STEP6_TRAIN_SCENES) & set(STEP6_TEST_SCENES)), (
    "STEP6_TRAIN_SCENES and STEP6_TEST_SCENES must not overlap."
)

assert len(STEP6_TRAIN_SCENES) == 90, (
    f"Expected 90 train scenes for heldout-bathroom split, got {len(STEP6_TRAIN_SCENES)}"
)

assert len(STEP6_TEST_SCENES) == 30, (
    f"Expected 30 test scenes for heldout-bathroom split, got {len(STEP6_TEST_SCENES)}"
)

STEP6_TRAIN_SCENE_TYPES_RESOLVED = [
    "bedroom",
    "kitchen",
    "living_room",
]

STEP6_TEST_SCENE_TYPES_RESOLVED = [
    "bathroom",
]

STEP6_REQUIRE_EXPLICIT_SCENE_SPLIT = True


# Step6 label-space preset.
# "all_directional": in/contains, on/supports, left/right, above/below
# "spatial_directional_only": left/right and above/below only

if STEP6_LABEL_SUBSET_MODE == "all_directional":
    STEP6_ALLOWED_LABELS = set(DIRECTIONAL_RELATIONS)

elif STEP6_LABEL_SUBSET_MODE == "spatial_directional_only":
    STEP6_ALLOWED_LABELS = {
        "left_of",
        "right_of",
        "above",
        "below",
    }

else:
    raise ValueError(
        f"Unsupported STEP6_LABEL_SUBSET_MODE: {STEP6_LABEL_SUBSET_MODE!r}"
    )

# Direct-only experiment:
#   only explicit_direct samples are kept.

STEP6_TRAIN_ONE_DIRECTION_PER_PAIR_GROUP = False
STEP6_TRAIN_DIRECTION_SELECTION_MODE = "none"
STEP6_APPLY_DIRECTION_FILTER_TO_TRAIN_ONLY = True
STEP6_DIRECTION_FILTER_GROUP_KEY = "pair_group_id"


# Linear probe hyperparameters

STEP6_LOGREG_MAX_ITER = 5000
STEP6_LOGREG_C = 1.0
STEP6_LOGREG_CLASS_WEIGHT = "balanced"
STEP6_LOGREG_SOLVER = "lbfgs"


# Output / diagnostic controls

STEP6_PRINT_DATASET_SUMMARY = True
STEP6_PRINT_LAYER_PROGRESS = True
STEP6_PRINT_TOP_LAYERS = True
STEP6_NUM_TOP_LAYERS_TO_PRINT = 10


# Local /content cache controls

STEP6_USE_LOCAL_CONTENT_CACHE = True

STEP6_LOCAL_CONTENT_CACHE_ROOT = os.path.join(
    "/content",
    f"step5_hs_cache_{STEP6_MODEL_TAG}_{STEP6_EXPERIMENT_SHORT_NAME}",
)

STEP6_LOCAL_CACHE_SAFETY_MARGIN_GB = 20.0

STEP6_LAYER_INDICES_TO_RUN = None
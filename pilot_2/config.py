import os

# pilot 根目录
PILOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(PILOT_DIR, "data")

# ===== Step1–3 =====
STEP1_DIR = os.path.join(DATA_DIR, "step1_ground_truth")
STEP2_DIR = os.path.join(DATA_DIR, "step2_relations")
STEP3_DIR = os.path.join(DATA_DIR, "step3_text")

# ===== Step4（LLM1）=====
STEP4_DIR = os.path.join(DATA_DIR, "step4_text_to_paragraph")

STEP4_OLMO_DIR = os.path.join(STEP4_DIR, "step4_text_refine_with_olmo2")
STEP4_GEMMA_DIR = os.path.join(STEP4_DIR, "step4_text_refine_with_gemma")
STEP4_LLAMA_DIR = os.path.join(STEP4_DIR, "step4_text_refine_with_llama")

# ===== 实验参数 =====
SCENES = ["FloorPlan1", "FloorPlan2"]
MAX_OBJECTS_PER_SCENE = 12
USE_PICKUPABLE_ONLY = False
RANDOM_SEED = 42
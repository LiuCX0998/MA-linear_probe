import os

# pilot/ 目录
PILOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(PILOT_DIR, "data")
STEP1_DIR = os.path.join(DATA_DIR, "step1_ground_truth")
STEP2_DIR = os.path.join(DATA_DIR, "step2_relations")
STEP3_DIR = os.path.join(DATA_DIR, "step3_text")

# 小规模 pilot：先只跑两个场景
SCENES = ["FloorPlan1", "FloorPlan2"]

# 每个场景最多保留多少对象
MAX_OBJECTS_PER_SCENE = 12

# 是否只保留 pickupable 物体
USE_PICKUPABLE_ONLY = False

# 随机种子
RANDOM_SEED = 42
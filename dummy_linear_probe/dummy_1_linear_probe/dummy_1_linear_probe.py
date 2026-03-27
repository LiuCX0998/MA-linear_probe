import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# 1. Global config
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OBJECT_VOCAB = [
    "apple", "microwave", "table", "chair", "fridge",
    "cabinet", "plate", "cup", "sofa", "lamp"
]

ROOM_TYPES = ["kitchen", "living_room", "office"]

NUM_SCENES = 80
OBJECTS_PER_SCENE_MIN = 4
OBJECTS_PER_SCENE_MAX = 7

HIDDEN_DIM = 32
CATEGORY_EMB_SCALE = 0.8
NOISE_SCALE = 0.15
SPACE_SIGNAL_SCALE = 1.0

ROOM_EMB_SCALE = 0.5
ROOM_COMPONENT_SCALE = 0.4
SCENE_EMB_SAMPLE_SCALE = 0.25
SCENE_CONTEXT_SCALE = 0.5
CONTEXT_MIX_SCALE = 0.3
RELATION_BIAS_SCALE = 0.25


TRAIN_RATIO = 0.8
RIDGE_ALPHA = 1.0


# =========================
# 2. Data structures
# =========================
@dataclass
class ObjectRecord:
    scene_id: int
    object_id: int
    object_name: str
    room_type: str
    x: float
    z: float
    parent_object_name: Optional[str]
    relation_tags: List[str]
    hidden: np.ndarray

@dataclass
class PairRecord:
    scene_id: int
    obj_i_name: str
    obj_j_name: str
    feature: np.ndarray
    target: np.ndarray


# =========================
# 3. Dummy dataset generator
# =========================
class DummySpatialDataset:
    def __init__(
            self,
            hidden_dim: int = HIDDEN_DIM,
            object_vocab: List[str] = OBJECT_VOCAB,
            room_types: List[str] = ROOM_TYPES,
            space_signal_scale: float = SPACE_SIGNAL_SCALE,
            category_emb_scale: float = CATEGORY_EMB_SCALE,
            room_emb_scale: float = ROOM_EMB_SCALE,
            room_component_scale: float = ROOM_COMPONENT_SCALE,
            scene_emb_sample_scale: float = SCENE_EMB_SAMPLE_SCALE,
            noise_scale: float = NOISE_SCALE,
            scene_context_scale: float = SCENE_CONTEXT_SCALE,
            context_mix_scale: float = CONTEXT_MIX_SCALE,
            relation_bias_scale: float = RELATION_BIAS_SCALE,
            random_seed: int = RANDOM_SEED,
    ):
        self.hidden_dim = hidden_dim
        self.object_vocab = object_vocab
        self.room_types = room_types
        self.space_signal_scale = space_signal_scale
        self.category_emb_scale = category_emb_scale
        self.room_emb_scale = room_emb_scale
        self.room_component_scale = room_component_scale
        self.scene_emb_sample_scale = scene_emb_sample_scale
        self.noise_scale = noise_scale
        self.scene_context_scale = scene_context_scale
        self.context_mix_scale = context_mix_scale
        self.relation_bias_scale = relation_bias_scale
        self.rng = np.random.default_rng(random_seed)

        # 2D coordinate -> hidden linear map
        self.coord_to_hidden = self.rng.normal(
            loc=0.0, scale=space_signal_scale, size=(2, hidden_dim)
        )

        # Restrict spatial information to a subset of dimensions to simulate a local subspace.
        self.space_mask = np.zeros(hidden_dim, dtype=np.float32)
        self.space_mask[8:16] = 1.0
        self.space_mask[24:32] = 1.0

        # category embeddings
        self.category_embeddings = {
            obj_name: self.rng.normal(loc=0.0, scale=category_emb_scale, size=(hidden_dim,))
            for obj_name in object_vocab
        }

        # room-type embeddings
        self.room_type_embeddings = {
            room: self.rng.normal(loc=0.0, scale=room_emb_scale, size=(hidden_dim,))
            for room in room_types
        }

        # relation-bias embeddings（Simulate the bias that relational words leave in the hidden states）
        self.relation_bias_embeddings = {
            "left": self.rng.normal(loc=0.0, scale=0.3, size=(hidden_dim,)),
            "right": self.rng.normal(loc=0.0, scale=0.3, size=(hidden_dim,)),
            "near": self.rng.normal(loc=0.0, scale=0.3, size=(hidden_dim,)),
            "on": self.rng.normal(loc=0.0, scale=0.3, size=(hidden_dim,)),
            "in": self.rng.normal(loc=0.0, scale=0.3, size=(hidden_dim,)),
        }

        # room -> candidate objects
        self.room_object_priors = {
            "kitchen": ["apple", "microwave", "table", "fridge", "cabinet", "plate", "cup", "chair"],
            "living_room": ["sofa", "lamp", "table", "chair", "cup"],
            "office": ["chair", "table", "lamp", "cabinet", "cup"],
        }

        # object-specific position prior centers
        self.object_position_priors = {
            "fridge": np.array([-3.5, 0.0]),
            "cabinet": np.array([3.0, 0.5]),
            "table": np.array([0.0, 0.0]),
            "microwave": np.array([2.3, 0.2]),
            "sofa": np.array([-2.5, -2.0]),
            "lamp": np.array([3.0, -3.0]),
            "chair": np.array([0.8, -0.8]),
            "plate": np.array([0.0, 0.1]),
            "cup": np.array([0.3, 0.2]),
            "apple": np.array([-0.2, 0.1]),
        }

    def _sample_room_type_and_objects(self, n_objects: int) -> Tuple[str, List[str]]:
        room_type = self.rng.choice(self.room_types)
        candidates = self.room_object_priors[room_type]

        n_objects = min(n_objects, len(candidates))
        chosen_objects = self.rng.choice(candidates, size=n_objects, replace=False).tolist()
        return room_type, chosen_objects

    def _sample_object_position(
            self,
            obj_name: str,
            object_positions: Dict[str, Tuple[float, float]],
    ) -> Tuple[float, float, Optional[str]]:
        parent_object_name = None

        if obj_name == "apple":
            if "fridge" in object_positions and self.rng.random() < 0.3:
                parent_object_name = "fridge"
                px, pz = object_positions["fridge"]
                x = px + self.rng.normal(0.0, 0.25)
                z = pz + self.rng.normal(0.0, 0.25)
                return float(x), float(z), parent_object_name

            if "table" in object_positions:
                parent_object_name = "table"
                px, pz = object_positions["table"]
                x = px + self.rng.normal(0.0, 0.35)
                z = pz + self.rng.normal(0.0, 0.35)
                return float(x), float(z), parent_object_name

        if obj_name in ["plate", "cup"] and "table" in object_positions:
            parent_object_name = "table"
            px, pz = object_positions["table"]
            x = px + self.rng.normal(0.0, 0.35)
            z = pz + self.rng.normal(0.0, 0.35)
            return float(x), float(z), parent_object_name

        center = self.object_position_priors[obj_name]
        x = self.rng.normal(center[0], 0.8)
        z = self.rng.normal(center[1], 0.8)
        return float(x), float(z), parent_object_name

    def _infer_relation_tags(
            self,
            current_spec: Dict[str, Any],
            all_specs: List[Dict[str, Any]],
    ) -> List[str]:
        tags = []

        parent = current_spec["parent_object_name"]
        if parent is not None:
            if parent in ["fridge", "cabinet"]:
                tags.append("in")
            else:
                tags.append("on")

        x_i, z_i = current_spec["x"], current_spec["z"]

        for other in all_specs:
            if other["object_id"] == current_spec["object_id"]:
                continue

            dx = other["x"] - x_i
            dz = other["z"] - z_i
            dist = np.sqrt(dx * dx + dz * dz)

            if dx > 1.0:
                tags.append("right")
            elif dx < -1.0:
                tags.append("left")

            if dist < 1.2:
                tags.append("near")

        return sorted(list(set(tags)))

    def _build_hidden(
            self,
            obj_name: str,
            x: float,
            z: float,
            room_type: str,
            relation_tags: List[str],
            scene_embedding: np.ndarray,
            scene_semantic_mean: np.ndarray,
            spatial_signal: bool,
    ) -> np.ndarray:
        coord_vec = np.array([x, z], dtype=np.float32)

        if spatial_signal:
            raw_spatial = coord_vec @ self.coord_to_hidden
            spatial_component = raw_spatial * self.space_mask
        else:
            spatial_component = np.zeros(self.hidden_dim, dtype=np.float32)

        semantic_component = self.category_embeddings[obj_name]
        room_component = self.room_component_scale * self.room_type_embeddings[room_type]
        scene_component = self.scene_context_scale * scene_embedding
        context_component = self.context_mix_scale * scene_semantic_mean

        relation_component = np.zeros(self.hidden_dim, dtype=np.float32)
        for tag in relation_tags:
            relation_component += self.relation_bias_embeddings[tag]
        relation_component *= self.relation_bias_scale

        noise_component = self.rng.normal(
            loc=0.0,
            scale=self.noise_scale,
            size=(self.hidden_dim,)
        )

        hidden = (
                spatial_component
                + semantic_component
                + room_component
                + scene_component
                + context_component
                + relation_component
                + noise_component
        )
        return hidden.astype(np.float32)

    def sample_scene(self, scene_id: int, n_objects: int, spatial_signal: bool = True) -> List[ObjectRecord]:
        room_type, chosen_objects = self._sample_room_type_and_objects(n_objects=n_objects)

        object_specs = []
        object_positions: Dict[str, Tuple[float, float]] = {}

        # pass 1: sample scene structure and coordinates
        for obj_id, obj_name in enumerate(chosen_objects):
            x, z, parent_object_name = self._sample_object_position(
                obj_name=obj_name,
                object_positions=object_positions,
            )
            object_positions[obj_name] = (x, z)
            object_specs.append({
                "object_id": obj_id,
                "object_name": obj_name,
                "x": x,
                "z": z,
                "parent_object_name": parent_object_name,
            })

        # scene-level shared context
        scene_embedding = self.rng.normal(
            loc=0.0,
            scale=self.scene_emb_sample_scale,
            size=(self.hidden_dim,)
        )

        scene_semantic_mean = np.mean(
            [self.category_embeddings[obj_name] for obj_name in chosen_objects],
            axis=0
        )

        # pass 2: infer relation tags and build hidden
        records = []
        for spec in object_specs:
            relation_tags = self._infer_relation_tags(
                current_spec=spec,
                all_specs=object_specs,
            )

            hidden = self._build_hidden(
                obj_name=spec["object_name"],
                x=spec["x"],
                z=spec["z"],
                room_type=room_type,
                relation_tags=relation_tags,
                scene_embedding=scene_embedding,
                scene_semantic_mean=scene_semantic_mean,
                spatial_signal=spatial_signal,
            )

            records.append(
                ObjectRecord(
                    scene_id=scene_id,
                    object_id=spec["object_id"],
                    object_name=spec["object_name"],
                    room_type=room_type,
                    x=float(spec["x"]),
                    z=float(spec["z"]),
                    parent_object_name=spec["parent_object_name"],
                    relation_tags=relation_tags,
                    hidden=hidden,
                )
            )

        return records

    def sample_dataset(
        self,
        num_scenes: int,
        objects_per_scene_min: int,
        objects_per_scene_max: int,
        spatial_signal: bool = True,
    ) -> List[ObjectRecord]:
        all_objects = []
        for scene_id in range(num_scenes):
            n_objects = self.rng.integers(objects_per_scene_min, objects_per_scene_max + 1)
            all_objects.extend(
                self.sample_scene(scene_id, n_objects, spatial_signal=spatial_signal)
            )
        return all_objects


# =========================
# 4. Pair feature construction
# =========================
def build_pair_feature(h_i: np.ndarray, h_j: np.ndarray) -> np.ndarray:
    """
    Pair feature = [h_i ; h_j ; h_j - h_i]
    """
    return np.concatenate([h_i, h_j, h_j - h_i], axis=0)


def build_pair_dataset(objects: List[ObjectRecord]) -> List[PairRecord]:
    by_scene: Dict[int, List[ObjectRecord]] = {}
    for obj in objects:
        by_scene.setdefault(obj.scene_id, []).append(obj)

    pair_records: List[PairRecord] = []
    for scene_id, scene_objects in by_scene.items():
        n = len(scene_objects)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                obj_i = scene_objects[i]
                obj_j = scene_objects[j]

                feature = build_pair_feature(obj_i.hidden, obj_j.hidden)
                target = np.array(
                    [obj_j.x - obj_i.x, obj_j.z - obj_i.z],
                    dtype=np.float32
                )

                pair_records.append(
                    PairRecord(
                        scene_id=scene_id,
                        obj_i_name=obj_i.object_name,
                        obj_j_name=obj_j.object_name,
                        feature=feature.astype(np.float32),
                        target=target,
                    )
                )
    return pair_records


# =========================
# 5. Split by scene
# =========================
def split_by_scene(pair_records: List[PairRecord], train_ratio: float = TRAIN_RATIO):
    scene_ids = sorted(set(p.scene_id for p in pair_records))
    n_train = int(len(scene_ids) * train_ratio)

    train_scene_ids = scene_ids[:n_train]
    test_scene_ids = scene_ids[n_train:]

    train_scene_set = set(train_scene_ids)
    test_scene_set = set(test_scene_ids)

    train_pairs = [p for p in pair_records if p.scene_id in train_scene_set]
    test_pairs = [p for p in pair_records if p.scene_id in test_scene_set]

    split_info = {
        "train_scene_ids": train_scene_ids,
        "test_scene_ids": test_scene_ids,
        "n_train_scenes": len(train_scene_ids),
        "n_test_scenes": len(test_scene_ids),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
    }
    return train_pairs, test_pairs, split_info


def pair_records_to_arrays(pair_records: List[PairRecord]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack([p.feature for p in pair_records], axis=0)
    Y = np.stack([p.target for p in pair_records], axis=0)
    return X, Y


# =========================
# 6. Training / evaluation separation
# =========================
def train_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    config: Dict[str, Any],
):
    """
    Return:
        model                -> trained sklearn model
        training_artifact    -> model-related outputs
        training_diagnostics -> train metrics
    """
    alpha = config.get("ridge_alpha", 1.0)
    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train)

    # training predictions
    Y_train_pred = model.predict(X_train)

    training_diagnostics = compute_metrics(
        y_true=Y_train,
        y_pred=Y_train_pred,
        split_name="train"
    )

    training_artifact = {
        "model_type": "Ridge",
        "ridge_alpha": alpha,
        "feature_dim": int(X_train.shape[1]),
        "target_dim": int(Y_train.shape[1]),
        "coef_shape": list(model.coef_.shape),
        "intercept_shape": list(np.atleast_1d(model.intercept_).shape),
        "coef": model.coef_.tolist(),
        "intercept": np.atleast_1d(model.intercept_).tolist(),
        "config": config,
    }

    return model, training_artifact, training_diagnostics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, split_name: str):
    mse_overall = mean_squared_error(y_true, y_pred)
    r2_overall = r2_score(y_true, y_pred, multioutput="uniform_average")

    mse_dx = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_dz = mean_squared_error(y_true[:, 1], y_pred[:, 1])

    r2_dx = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_dz = r2_score(y_true[:, 1], y_pred[:, 1])

    residuals = y_true - y_pred
    residual_norm = np.linalg.norm(residuals, axis=1)

    metrics = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "mse_overall": float(mse_overall),
        "r2_overall": float(r2_overall),
        "mse_dx": float(mse_dx),
        "mse_dz": float(mse_dz),
        "r2_dx": float(r2_dx),
        "r2_dz": float(r2_dz),
        "mean_residual_norm": float(np.mean(residual_norm)),
        "std_residual_norm": float(np.std(residual_norm)),
    }
    return metrics


def evaluate_probe(model, X: np.ndarray, Y: np.ndarray, split_name: str = "test"):
    """
    Return:
        evaluation_result -> metrics + predictions + residual summaries
    """
    Y_pred = model.predict(X)
    metrics = compute_metrics(Y, Y_pred, split_name=split_name)

    evaluation_result = {
        "metrics": metrics,
        "y_true": Y.tolist(),
        "y_pred": Y_pred.tolist(),
        "residuals": (Y - Y_pred).tolist(),
    }
    return evaluation_result


# =========================
# 7. Print
# =========================
def print_metrics(metrics: Dict[str, Any], title: str):
    print(f"\n===== {title} =====")
    print(f"split: {metrics['split']}")
    print(f"n_samples: {metrics['n_samples']}")
    print(f"MSE overall: {metrics['mse_overall']:.4f}")
    print(f"R2  overall: {metrics['r2_overall']:.4f}")
    print(f"MSE dx: {metrics['mse_dx']:.4f}")
    print(f"MSE dz: {metrics['mse_dz']:.4f}")
    print(f"R2 dx:  {metrics['r2_dx']:.4f}")
    print(f"R2 dz:  {metrics['r2_dz']:.4f}")
    print(f"mean residual norm: {metrics['mean_residual_norm']:.4f}")
    print(f"std residual norm:  {metrics['std_residual_norm']:.4f}")


def print_training_artifact_summary(training_artifact: Dict[str, Any], split_info: Dict[str, Any]):
    print("\n===== Training Artifact Summary =====")
    print(f"Model type: {training_artifact['model_type']}")
    print(f"Ridge alpha: {training_artifact['ridge_alpha']}")
    print(f"Feature dim: {training_artifact['feature_dim']}")
    print(f"Target dim: {training_artifact['target_dim']}")
    print(f"Coef shape: {training_artifact['coef_shape']}")
    print(f"Intercept shape: {training_artifact['intercept_shape']}")
    print(f"Train scenes: {split_info['n_train_scenes']} | Test scenes: {split_info['n_test_scenes']}")
    print(f"Train pairs: {split_info['n_train_pairs']} | Test pairs: {split_info['n_test_pairs']}")


def print_sample_objects(objects: List[ObjectRecord], n: int = 5):
    print("\n===== Sample Object Records =====")
    for idx, obj in enumerate(objects[:n]):
        print(
            f"[{idx}] scene={obj.scene_id}, room={obj.room_type}, "
            f"obj={obj.object_name}, parent={obj.parent_object_name}, "
            f"pos=({obj.x:.2f}, {obj.z:.2f}), tags={obj.relation_tags}"
        )

def print_sample_pairs(pair_records: List[PairRecord], n: int = 3):
    print("\n===== Sample Pair Records =====")
    for idx, p in enumerate(pair_records[:n]):
        print(f"[{idx}] scene={p.scene_id}, {p.obj_i_name} -> {p.obj_j_name}")
        print(f"     feature shape: {p.feature.shape}")
        print(f"     target [dx, dz]: {p.target}")


# =========================
# 8. One experiment runner
# =========================
def run_one_experiment(
    spatial_signal: bool,
    exp_name: str,
    save_json: bool = False,
):
    dataset = DummySpatialDataset(
        hidden_dim=HIDDEN_DIM,
        object_vocab=OBJECT_VOCAB,
        room_types=ROOM_TYPES,
        space_signal_scale=SPACE_SIGNAL_SCALE,
        category_emb_scale=CATEGORY_EMB_SCALE,
        room_emb_scale=ROOM_EMB_SCALE,
        room_component_scale=ROOM_COMPONENT_SCALE,
        scene_emb_sample_scale=SCENE_EMB_SAMPLE_SCALE,
        noise_scale=NOISE_SCALE,
        scene_context_scale=SCENE_CONTEXT_SCALE,
        context_mix_scale=CONTEXT_MIX_SCALE,
        relation_bias_scale=RELATION_BIAS_SCALE,
        random_seed=RANDOM_SEED,
    )

    objects = dataset.sample_dataset(
        num_scenes=NUM_SCENES,
        objects_per_scene_min=OBJECTS_PER_SCENE_MIN,
        objects_per_scene_max=OBJECTS_PER_SCENE_MAX,
        spatial_signal=spatial_signal,
    )

    pairs = build_pair_dataset(objects)
    train_pairs, test_pairs, split_info = split_by_scene(pairs, train_ratio=TRAIN_RATIO)

    X_train, Y_train = pair_records_to_arrays(train_pairs)
    X_test, Y_test = pair_records_to_arrays(test_pairs)

    config = {
        "experiment_name": exp_name,
        "random_seed": RANDOM_SEED,
        "hidden_dim": HIDDEN_DIM,
        "feature_type": "[h_i; h_j; h_j-h_i]",
        "target_type": "relative_2d_coords_(dx,dz)",
        "ridge_alpha": RIDGE_ALPHA,
        "train_ratio": TRAIN_RATIO,
        "spatial_signal": spatial_signal,
        "num_scenes": NUM_SCENES,
        "objects_per_scene_min": OBJECTS_PER_SCENE_MIN,
        "objects_per_scene_max": OBJECTS_PER_SCENE_MAX,
        "room_types": ROOM_TYPES,
        "room_emb_scale": ROOM_EMB_SCALE,
        "room_component_scale": ROOM_COMPONENT_SCALE,
        "scene_emb_sample_scale": SCENE_EMB_SAMPLE_SCALE,
        "scene_context_scale": SCENE_CONTEXT_SCALE,
        "context_mix_scale": CONTEXT_MIX_SCALE,
        "relation_bias_scale": RELATION_BIAS_SCALE,
    }

    model, training_artifact, training_diagnostics = train_probe(
        X_train=X_train,
        Y_train=Y_train,
        config=config,
    )

    evaluation_result = evaluate_probe(
        model=model,
        X=X_test,
        Y=Y_test,
        split_name="test"
    )

    experiment_output = {
        "experiment_name": exp_name,
        "config": config,
        "split_info": split_info,
        "training_artifact": training_artifact,
        "training_diagnostics": training_diagnostics,
        "evaluation_result": evaluation_result,
    }

    print(f"\n==============================")
    print(f"Experiment: {exp_name}")
    print(f"Spatial signal present: {spatial_signal}")
    print(f"Num objects: {len(objects)}")
    print(f"Num pairs: {len(pairs)}")

    print_sample_objects(objects, n=5)
    print_sample_pairs(train_pairs, n=3)
    print_training_artifact_summary(training_artifact, split_info)
    print_metrics(training_diagnostics, title="Training Diagnostics")
    print_metrics(evaluation_result["metrics"], title="Evaluation Result (Test)")


    y_true = np.array(evaluation_result["y_true"])
    y_pred = np.array(evaluation_result["y_pred"])


    if save_json:
        saveable_output = make_json_safe(experiment_output)
        filename = exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(saveable_output, f, ensure_ascii=False, indent=2)
        print(f"\nSaved experiment output to: {filename}")

    return experiment_output


# =========================
# 9. JSON utility
# =========================
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


# =========================
# 10. Main
# =========================
if __name__ == "__main__":
    positive_output = run_one_experiment(
        spatial_signal=True,
        exp_name="Positive Control (Spatial Signal Present)",
        save_json=False,
    )

    negative_output = run_one_experiment(
        spatial_signal=False,
        exp_name="Negative Control (No Spatial Signal)",
        save_json=False,
    )

    print("\n==============================")
    print("Quick comparison on TEST R²:")
    print(
        f"Positive test R²: "
        f"{positive_output['evaluation_result']['metrics']['r2_overall']:.4f}"
    )
    print(
        f"Negative test R²: "
        f"{negative_output['evaluation_result']['metrics']['r2_overall']:.4f}"
    )

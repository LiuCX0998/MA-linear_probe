import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
#import matplotlib.pyplot as plt


# =========================
# 1. Global config
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OBJECT_VOCAB = [
    "apple", "microwave", "table", "chair", "fridge",
    "cabinet", "plate", "cup", "sofa", "lamp"
]

NUM_SCENES = 80
OBJECTS_PER_SCENE_MIN = 4
OBJECTS_PER_SCENE_MAX = 7

HIDDEN_DIM = 32
CATEGORY_EMB_SCALE = 0.8
NOISE_SCALE = 0.15
SPACE_SIGNAL_SCALE = 1.0

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
    x: float
    z: float
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
        space_signal_scale: float = SPACE_SIGNAL_SCALE,
        category_emb_scale: float = CATEGORY_EMB_SCALE,
        noise_scale: float = NOISE_SCALE,
        random_seed: int = RANDOM_SEED,
    ):
        self.hidden_dim = hidden_dim
        self.object_vocab = object_vocab
        self.space_signal_scale = space_signal_scale
        self.category_emb_scale = category_emb_scale
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(random_seed)

        # 2D coordinate -> hidden space linear map
        self.coord_to_hidden = self.rng.normal(
            loc=0.0, scale=space_signal_scale, size=(2, hidden_dim)
        )

        # category embeddings
        self.category_embeddings = {
            obj_name: self.rng.normal(loc=0.0, scale=category_emb_scale, size=(hidden_dim,))
            for obj_name in object_vocab
        }

    def sample_scene(self, scene_id: int, n_objects: int, spatial_signal: bool = True) -> List[ObjectRecord]:
        chosen_objects = self.rng.choice(self.object_vocab, size=n_objects, replace=False)
        records = []

        for obj_id, obj_name in enumerate(chosen_objects):
            x = self.rng.uniform(-5.0, 5.0)
            z = self.rng.uniform(-5.0, 5.0)
            coord_vec = np.array([x, z])

            if spatial_signal:
                hidden = coord_vec @ self.coord_to_hidden
            else:
                hidden = np.zeros(self.hidden_dim)

            hidden = (
                hidden
                + self.category_embeddings[obj_name]
                + self.rng.normal(loc=0.0, scale=self.noise_scale, size=(self.hidden_dim,))
            )

            records.append(
                ObjectRecord(
                    scene_id=scene_id,
                    object_id=obj_id,
                    object_name=obj_name,
                    x=float(x),
                    z=float(z),
                    hidden=hidden.astype(np.float32),
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

    pair_records = []
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
# 7. Pretty printing
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


def print_sample_pairs(pair_records: List[PairRecord], n: int = 3):
    print("\n===== Sample Pair Records =====")
    for idx, p in enumerate(pair_records[:n]):
        print(f"[{idx}] scene={p.scene_id}, {p.obj_i_name} -> {p.obj_j_name}")
        print(f"     feature shape: {p.feature.shape}")
        print(f"     target [dx, dz]: {p.target}")


# =========================
# 8. Plotting
# =========================
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    mn = min(y_true[:, 0].min(), y_pred[:, 0].min())
    mx = max(y_true[:, 0].max(), y_pred[:, 0].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True dx")
    plt.ylabel("Predicted dx")
    plt.title(f"{title_prefix} - dx")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    mn = min(y_true[:, 1].min(), y_pred[:, 1].min())
    mx = max(y_true[:, 1].max(), y_pred[:, 1].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True dz")
    plt.ylabel("Predicted dz")
    plt.title(f"{title_prefix} - dz")
    plt.tight_layout()
    plt.show()


# =========================
# 9. One experiment runner
# =========================
def run_one_experiment(
    spatial_signal: bool,
    exp_name: str,
    save_json: bool = False,
):
    dataset = DummySpatialDataset(
        hidden_dim=HIDDEN_DIM,
        object_vocab=OBJECT_VOCAB,
        space_signal_scale=SPACE_SIGNAL_SCALE,
        category_emb_scale=CATEGORY_EMB_SCALE,
        noise_scale=NOISE_SCALE,
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

    print_sample_pairs(train_pairs, n=3)
    print_training_artifact_summary(training_artifact, split_info)
    print_metrics(training_diagnostics, title="Training Diagnostics")
    print_metrics(evaluation_result["metrics"], title="Evaluation Result (Test)")

    y_true = np.array(evaluation_result["y_true"])
    y_pred = np.array(evaluation_result["y_pred"])
    #plot_predictions(y_true, y_pred, title_prefix=exp_name)

    if save_json:
        saveable_output = make_json_safe(experiment_output)
        filename = exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(saveable_output, f, ensure_ascii=False, indent=2)
        print(f"\nSaved experiment output to: {filename}")

    return experiment_output


# =========================
# 10. JSON utility
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
# 11. Main
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
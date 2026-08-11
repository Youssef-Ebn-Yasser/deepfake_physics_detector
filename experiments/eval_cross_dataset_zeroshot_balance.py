"""
Zero-Shot Cross-Dataset Evaluation

Loads the BEST trained Stage D model and evaluates it WITHOUT
any retraining on external datasets.

Training:
    DeeperForensics-1.0

External evaluation:
    1. Celeb-DF v2
    2. FaceForensics++

IMPORTANT:
    - Model is frozen.
    - No training occurs.
    - No fine-tuning occurs.
    - No scaler fitting occurs.
    - The scaler fitted on DeeperForensics training data is reused.
    - External test data can optionally be balanced.

Expected feature structure:

DataSets/
└── cross_dataset_features/
    ├── celeb_df/
    │   ├── real/
    │   │   └── *.npz
    │   ├── fake/
    │   │   └── *.npz
    │   └── manifest.csv
    │
    └── ff_plus_plus/
        ├── real/
        │   └── *.npz
        ├── fake/
        │   └── *.npz
        └── manifest.csv

Each .npz must contain:

    f1
    f2
    f3
    f4
    f5

Usage:

    python experiments/eval_cross_dataset_zeroshot.py
"""

import os
import sys
import json

import numpy as np
import tensorflow as tf


# ============================================================================
# PATH SETUP
# ============================================================================

# Make project root importable BEFORE importing local modules
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# CUSTOM MODEL COMPONENTS
# ============================================================================

from stage_d_full_model import (
    TransformerEncoderBlock,
    CrossAttentionBlock,
    GatedFusion,
)


# ============================================================================
# SHARED UTILITIES
# ============================================================================

from experiments.shared_utils import (
    FEATURE_KEYS,
    FeatureScaler,
    compute_all_metrics,
    print_metrics,
    save_metrics,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Stage D directory
STAGE_D_DIR = os.path.join(
    "checkpoints",
    "deepfakes",
    "stage_d"
)

# Frozen model
MODEL_PATH = os.path.join(
    STAGE_D_DIR,
    "best_model.keras"
)

# IMPORTANT:
# This scaler MUST be the scaler fitted on the DeeperForensics
# TRAINING DATA.
SCALER_PATH = os.path.join(
    STAGE_D_DIR,
    "scaler.pkl"
)


# ============================================================================
# CROSS-DATASET FEATURE DIRECTORIES
# ============================================================================

CROSS_DATASET_BASE = os.path.join(
    "DataSets",
    "cross_dataset_features"
)


DATASETS = {
    "Celeb-DF v2": os.path.join(
        CROSS_DATASET_BASE,
        "celeb_df"
    ),

    "FaceForensics++": os.path.join(
        CROSS_DATASET_BASE,
        "ff_plus_plus"
    ),
}


# ============================================================================
# EXTERNAL TEST BALANCING
# ============================================================================

# True  = balance this external test set
# False = keep original dataset distribution
#
# Recommended for your diagnostic:
#
# Celeb-DF-v2 -> balanced
# FF++        -> original
#
# You can change FF++ to True later.
BALANCE_EXTERNAL_TEST = {
    "Celeb-DF v2": True,
    "FaceForensics++": False,
}


# Reproducible sampling
RANDOM_SEED = 42


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = os.path.join(
    "checkpoints",
    "deepfakes",
    "cross_dataset"
)


# ============================================================================
# LOAD CROSS-DATASET
# ============================================================================

def load_cross_dataset(
    dataset_dir,
    dataset_name=None,
    balance=False,
    seed=42,
):
    """
    Load external .npz features.

    Labels are read from:

        manifest.csv

    or, if no manifest exists, from:

        real/
        fake/

    Label convention:

        0 = real
        1 = fake

    If balance=True:

        Number of real samples == number of fake samples.

    IMPORTANT:

        This function ONLY samples the TEST SET.

        It does NOT:
            - train
            - fine-tune
            - fit scaler
            - modify model
            - learn anything from external data
    """

    manifest_path = os.path.join(
        dataset_dir,
        "manifest.csv"
    )

    entries = []

    # ========================================================================
    # LOAD MANIFEST
    # ========================================================================

    if os.path.exists(manifest_path):

        print(
            f"  Loading manifest: {manifest_path}"
        )

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        if not lines:

            print("  [ERROR] manifest.csv is empty.")
            return None, None

        header = lines[0].strip().split(",")

        path_col = (
            header.index("npz_path")
            if "npz_path" in header
            else 0
        )

        label_col = (
            header.index("label")
            if "label" in header
            else 1
        )

        for line in lines[1:]:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) <= max(
                path_col,
                label_col
            ):
                continue

            npz_path = os.path.join(
                dataset_dir,
                parts[path_col].replace(
                    "/",
                    os.sep
                )
            )

            try:
                label = int(
                    parts[label_col]
                )
            except ValueError:
                continue

            entries.append(
                (
                    npz_path,
                    label
                )
            )

    # ========================================================================
    # AUTO-DISCOVER REAL / FAKE FOLDERS
    # ========================================================================

    else:

        print(
            "  No manifest.csv found."
        )

        print(
            "  Trying real/fake subfolders..."
        )

        label_folders = [
            ("real", 0),
            ("fake", 1),
            ("source", 0),
            ("manipulated", 1),
        ]

        for label_name, label_value in label_folders:

            subdir = os.path.join(
                dataset_dir,
                label_name
            )

            if not os.path.isdir(subdir):
                continue

            for root, _, files in os.walk(subdir):

                for fname in files:

                    if not fname.lower().endswith(
                        ".npz"
                    ):
                        continue

                    path = os.path.join(
                        root,
                        fname
                    )

                    entries.append(
                        (
                            path,
                            label_value
                        )
                    )

    # ========================================================================
    # CHECK ENTRIES
    # ========================================================================

    if not entries:

        print(
            "  [ERROR] No .npz samples found."
        )

        return None, None

    print(
        f"  Discovered {len(entries)} entries."
    )

    # ========================================================================
    # VALIDATE NPZ FILES
    # ========================================================================

    valid_entries = []

    missing = 0
    invalid = 0

    for npz_path, label in entries:

        if not os.path.exists(npz_path):

            missing += 1
            continue

        try:

            with np.load(
                npz_path,
                allow_pickle=False
            ) as data:

                if not all(
                    key in data
                    for key in FEATURE_KEYS
                ):

                    invalid += 1
                    continue

        except Exception as e:

            invalid += 1

            print(
                f"  [WARN] Invalid NPZ: "
                f"{npz_path}"
            )

            print(
                f"         {e}"
            )

            continue

        valid_entries.append(
            (
                npz_path,
                label
            )
        )

    if missing:

        print(
            f"  [WARN] Missing files: {missing}"
        )

    if invalid:

        print(
            f"  [WARN] Invalid/incomplete files: "
            f"{invalid}"
        )

    entries = valid_entries

    if not entries:

        print(
            "  [ERROR] No valid samples remain."
        )

        return None, None

    # ========================================================================
    # SEPARATE CLASSES
    # ========================================================================

    real_entries = [
        item
        for item in entries
        if item[1] == 0
    ]

    fake_entries = [
        item
        for item in entries
        if item[1] == 1
    ]

    print(
        f"  Original distribution:"
    )

    print(
        f"    Real: {len(real_entries)}"
    )

    print(
        f"    Fake: {len(fake_entries)}"
    )

    print(
        f"    Total: {len(entries)}"
    )

    # ========================================================================
    # BALANCE EXTERNAL TEST DATA
    # ========================================================================

    if balance:

        if len(real_entries) == 0:

            print(
                "  [ERROR] No real samples."
            )

            return None, None

        if len(fake_entries) == 0:

            print(
                "  [ERROR] No fake samples."
            )

            return None, None

        rng = np.random.default_rng(
            seed
        )

        n_each = min(
            len(real_entries),
            len(fake_entries)
        )

        # ------------------------------------------------------------
        # Randomly select real samples
        # ------------------------------------------------------------

        real_indices = rng.choice(
            len(real_entries),
            size=n_each,
            replace=False
        )

        # ------------------------------------------------------------
        # Randomly select fake samples
        # ------------------------------------------------------------

        fake_indices = rng.choice(
            len(fake_entries),
            size=n_each,
            replace=False
        )

        real_entries = [
            real_entries[i]
            for i in real_indices
        ]

        fake_entries = [
            fake_entries[i]
            for i in fake_indices
        ]

        entries = (
            real_entries +
            fake_entries
        )

        # ------------------------------------------------------------
        # Shuffle final balanced dataset
        # ------------------------------------------------------------

        rng.shuffle(entries)

        print()
        print(
            "  BALANCED EXTERNAL TEST SET"
        )

        print(
            f"    Real: {n_each}"
        )

        print(
            f"    Fake: {n_each}"
        )

        print(
            f"    Total: {2 * n_each}"
        )

        print(
            f"    Random seed: {seed}"
        )

    else:

        print()
        print(
            "  Using ORIGINAL class distribution."
        )

    # ========================================================================
    # LOAD FEATURES
    # ========================================================================

    X_dict = {
        key: []
        for key in FEATURE_KEYS
    }

    y_list = []

    failed_loads = 0

    for npz_path, label in entries:

        try:

            with np.load(
                npz_path,
                allow_pickle=False
            ) as data:

                for key in FEATURE_KEYS:

                    feature = data[key].astype(
                        np.float32
                    )

                    X_dict[key].append(
                        feature
                    )

                y_list.append(
                    label
                )

        except Exception as e:

            failed_loads += 1

            print(
                f"  [WARN] Failed to load:"
            )

            print(
                f"         {npz_path}"
            )

            print(
                f"         {e}"
            )

    if failed_loads:

        print(
            f"  [WARN] Failed loads: "
            f"{failed_loads}"
        )

    if not y_list:

        print(
            "  [ERROR] No samples loaded."
        )

        return None, None

    # ========================================================================
    # STACK FEATURES
    # ========================================================================

    for key in FEATURE_KEYS:

        try:

            X_dict[key] = np.stack(
                X_dict[key],
                axis=0
            )

        except Exception as e:

            print(
                f"  [ERROR] Could not stack "
                f"feature {key}: {e}"
            )

            return None, None

    y = np.asarray(
        y_list,
        dtype=np.int32
    )

    # ========================================================================
    # FINAL DISTRIBUTION
    # ========================================================================

    print()

    print(
        f"  Final loaded samples: "
        f"{len(y)}"
    )

    print(
        f"    Real: {(y == 0).sum()}"
    )

    print(
        f"    Fake: {(y == 1).sum()}"
    )

    # ========================================================================
    # FEATURE SHAPES
    # ========================================================================

    for key in FEATURE_KEYS:

        print(
            f"    {key}: "
            f"{X_dict[key].shape}"
        )

    return X_dict, y


# ============================================================================
# MODEL INPUT ORDER
# ============================================================================

def get_model_feature_order(model):
    """
    Determine the feature input order expected by the trained model.

    Example:

        [f1, f2, f3, f4, f5]
    """

    fk_order = []

    for inp in model.inputs:

        name = inp.name.lower()

        found = None

        for feature_key in FEATURE_KEYS:

            if feature_key.lower() in name:

                found = feature_key
                break

        if found is not None:

            fk_order.append(found)

    # ========================================================================
    # FALLBACK
    # ========================================================================

    if len(fk_order) != len(model.inputs):

        print(
            "[WARN] Could not determine model "
            "input order from input names."
        )

        print(
            "[WARN] Falling back to FEATURE_KEYS order."
        )

        fk_order = FEATURE_KEYS[
            :len(model.inputs)
        ]

    print()
    print(
        "Model input order:"
    )

    for i, key in enumerate(
        fk_order
    ):

        print(
            f"  Input {i}: {key}"
        )

    return fk_order


# ============================================================================
# PREDICTION
# ============================================================================

def predict_model(
    model,
    X_norm,
    input_order
):
    """
    Run frozen model prediction.

    No training.
    No gradient calculation.
    No weight updates.
    """

    test_X = [
        X_norm[key]
        for key in input_order
    ]

    raw_output = model.predict(
        test_X,
        verbose=0
    )

    # ========================================================================
    # HANDLE DICT OUTPUT
    # ========================================================================

    if isinstance(
        raw_output,
        dict
    ):

        if "logit" in raw_output:

            output = raw_output["logit"]

        elif "probability" in raw_output:

            output = raw_output[
                "probability"
            ]

        elif "output" in raw_output:

            output = raw_output["output"]

        else:

            # Use first output
            output = next(
                iter(
                    raw_output.values()
                )
            )

    # ========================================================================
    # HANDLE LIST OUTPUT
    # ========================================================================

    elif isinstance(
        raw_output,
        (list, tuple)
    ):

        output = raw_output[0]

    else:

        output = raw_output

    y_pred_proba = np.asarray(
        output
    ).reshape(-1)

    # ========================================================================
    # CHECK OUTPUT
    # ========================================================================

    if len(y_pred_proba) != len(
        X_norm[input_order[0]]
    ):

        raise ValueError(
            "Model output size does not "
            "match number of samples."
        )

    return y_pred_proba


# ============================================================================
# PREDICTION DIAGNOSTICS
# ============================================================================

def print_prediction_diagnostics(
    y_true,
    y_pred_proba,
    dataset_name
):
    """
    Print probability statistics.

    This is very useful for diagnosing
    cross-dataset domain shift.
    """

    print()
    print(
        f"Prediction diagnostics: "
        f"{dataset_name}"
    )

    real_probs = y_pred_proba[
        y_true == 0
    ]

    fake_probs = y_pred_proba[
        y_true == 1
    ]

    if len(real_probs):

        print(
            "  REAL probability:"
        )

        print(
            f"    mean   = "
            f"{np.mean(real_probs):.6f}"
        )

        print(
            f"    median = "
            f"{np.median(real_probs):.6f}"
        )

        print(
            f"    min    = "
            f"{np.min(real_probs):.6f}"
        )

        print(
            f"    max    = "
            f"{np.max(real_probs):.6f}"
        )

    if len(fake_probs):

        print(
            "  FAKE probability:"
        )

        print(
            f"    mean   = "
            f"{np.mean(fake_probs):.6f}"
        )

        print(
            f"    median = "
            f"{np.median(fake_probs):.6f}"
        )

        print(
            f"    min    = "
            f"{np.min(fake_probs):.6f}"
        )

        print(
            f"    max    = "
            f"{np.max(fake_probs):.6f}"
        )


# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

def save_prediction_arrays(
    dataset_name,
    y_true,
    y_pred_proba
):
    """
    Save predictions so they can be analyzed later.
    """

    safe_name = (
        dataset_name
        .replace(" ", "_")
        .replace("+", "_")
        .replace("-", "_")
        .lower()
    )

    path = os.path.join(
        OUTPUT_DIR,
        f"predictions_{safe_name}.npz"
    )

    np.savez_compressed(
        path,
        y_true=y_true,
        y_pred_proba=y_pred_proba
    )

    print(
        f"  Predictions saved: {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    # ========================================================================
    # HEADER
    # ========================================================================

    print()
    print("=" * 70)
    print(
        "ZERO-SHOT CROSS-DATASET EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        "Training dataset: DeeperForensics-1.0"
    )

    print(
        "External datasets:"
    )

    print(
        "  - Celeb-DF v2"
    )

    print(
        "  - FaceForensics++"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Model is FROZEN."
    )

    print(
        "  No retraining."
    )

    print(
        "  No fine-tuning."
    )

    print(
        "  No external scaler fitting."
    )

    print(
        f"  Random seed: {RANDOM_SEED}"
    )

    # ========================================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================================
    # CHECK MODEL
    # ========================================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print()
        print(
            f"[ERROR] Model not found:"
        )

        print(
            f"        {MODEL_PATH}"
        )

        print()

        print(
            "Expected:"
        )

        print(
            "  checkpoints/deepfakes/"
            "stage_d/best_model.keras"
        )

        sys.exit(1)

    # ========================================================================
    # CHECK SCALER
    # ========================================================================

    if not os.path.exists(
        SCALER_PATH
    ):

        print()
        print(
            f"[ERROR] Scaler not found:"
        )

        print(
            f"        {SCALER_PATH}"
        )

        sys.exit(1)

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    print()
    print(
        f"Loading frozen model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print()

    model = tf.keras.models.load_model(

        MODEL_PATH,

        compile=False,

        custom_objects={

            "TransformerEncoderBlock":
                TransformerEncoderBlock,

            "CrossAttentionBlock":
                CrossAttentionBlock,

            "GatedFusion":
                GatedFusion,
        }
    )

    print()
    print(
        "Model loaded successfully."
    )

    print(
        f"Model inputs: "
        f"{len(model.inputs)}"
    )

    # ========================================================================
    # FREEZE MODEL
    # ========================================================================

    # This does not train anything.
    # It makes the evaluation intent explicit.

    model.trainable = False

    for layer in model.layers:

        layer.trainable = False

    print(
        "Model frozen: YES"
    )

    # ========================================================================
    # MODEL INPUT ORDER
    # ========================================================================

    input_order = get_model_feature_order(
        model
    )

    # ========================================================================
    # LOAD TRAINING SCALER
    # ========================================================================

    print()
    print(
        f"Loading DeeperForensics "
        f"training scaler:"
    )

    print(
        f"  {SCALER_PATH}"
    )

    scaler = FeatureScaler.load(
        SCALER_PATH
    )

    print(
        "Training scaler loaded."
    )

    # ========================================================================
    # RESULTS
    # ========================================================================

    all_results = {}

    # ========================================================================
    # DATASET LOOP
    # ========================================================================

    for ds_name, ds_dir in DATASETS.items():

        print()
        print("=" * 70)

        print(
            f"Zero-Shot Evaluation: "
            f"{ds_name}"
        )

        print(
            f"Directory:"
        )

        print(
            f"  {ds_dir}"
        )

        print("=" * 70)

        # --------------------------------------------------------------------
        # Check directory
        # --------------------------------------------------------------------

        if not os.path.isdir(
            ds_dir
        ):

            print()
            print(
                "[SKIP] Directory does not exist."
            )

            print(
                f"       {ds_dir}"
            )

            all_results[
                ds_name
            ] = (
                "SKIPPED — features not found"
            )

            continue

        # --------------------------------------------------------------------
        # Determine balancing
        # --------------------------------------------------------------------

        balance_test = (
            BALANCE_EXTERNAL_TEST.get(
                ds_name,
                False
            )
        )

        print()

        print(
            f"Balance test set: "
            f"{'YES' if balance_test else 'NO'}"
        )

        # --------------------------------------------------------------------
        # Load dataset
        # --------------------------------------------------------------------

        X_dict, y_test = load_cross_dataset(

            ds_dir,

            dataset_name=ds_name,

            balance=balance_test,

            seed=RANDOM_SEED,
        )

        if X_dict is None:

            print()
            print(
                "[SKIP] No valid samples."
            )

            all_results[
                ds_name
            ] = (
                "SKIPPED — no samples"
            )

            continue

        # ====================================================================
        # APPLY TRAINING SCALER
        # ====================================================================

        print()
        print(
            "Applying DeeperForensics "
            "TRAINING scaler..."
        )

        print(
            "  Refit scaler: NO"
        )

        print(
            "  Fit on external data: NO"
        )

        X_norm = scaler.transform(
            X_dict
        )

        print(
            "  Transformation complete."
        )

        # ====================================================================
        # PREDICT
        # ====================================================================

        print()
        print(
            "Running frozen model prediction..."
        )

        y_pred_proba = predict_model(

            model,

            X_norm,

            input_order
        )

        print(
            "Prediction complete."
        )

        # ====================================================================
        # DIAGNOSTICS
        # ====================================================================

        print_prediction_diagnostics(

            y_test,

            y_pred_proba,

            ds_name
        )

        # ====================================================================
        # METRICS
        # ====================================================================

        metrics = compute_all_metrics(

            y_test,

            y_pred_proba
        )

        # ====================================================================
        # METADATA
        # ====================================================================

        metrics["dataset"] = ds_name

        metrics["model_src"] = MODEL_PATH

        metrics["scaler_src"] = SCALER_PATH

        metrics["n_samples"] = int(
            len(y_test)
        )

        metrics["n_real"] = int(
            (y_test == 0).sum()
        )

        metrics["n_fake"] = int(
            (y_test == 1).sum()
        )

        metrics["balanced_test"] = bool(
            balance_test
        )

        metrics["random_seed"] = (
            RANDOM_SEED
        )

        metrics["zero_shot"] = True

        # ====================================================================
        # PRINT METRICS
        # ====================================================================

        print()

        print_metrics(
            f"Zero-Shot: {ds_name}",
            metrics
        )

        # ====================================================================
        # SAVE METRICS
        # ====================================================================

        ds_safe = (
            ds_name
            .replace(" ", "_")
            .replace("+", "_")
            .replace("-", "_")
            .lower()
        )

        metrics_path = os.path.join(

            OUTPUT_DIR,

            f"metrics_{ds_safe}.json"
        )

        save_metrics(
            metrics,
            metrics_path
        )

        print(
            f"  Metrics saved:"
        )

        print(
            f"  {metrics_path}"
        )

        # ====================================================================
        # SAVE PREDICTIONS
        # ====================================================================

        save_prediction_arrays(

            ds_name,

            y_test,

            y_pred_proba
        )

        # ====================================================================
        # STORE
        # ====================================================================

        all_results[
            ds_name
        ] = metrics

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print()
    print()
    print("=" * 85)

    print(
        "ZERO-SHOT CROSS-DATASET "
        "EVALUATION SUMMARY"
    )

    print("=" * 85)

    print()

    print(
        f"{'Dataset':<22}"
        f"{'N':>8}"
        f"{'Real':>8}"
        f"{'Fake':>8}"
        f"{'Acc':>10}"
        f"{'F1':>10}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
    )

    print(
        "-" * 85
    )

    for ds_name, result in (
        all_results.items()
    ):

        if isinstance(
            result,
            str
        ):

            print(
                f"{ds_name:<22} "
                f"{result}"
            )

            continue

        print(
            f"{ds_name:<22}"
            f"{result['n_samples']:>8}"
            f"{result['n_real']:>8}"
            f"{result['n_fake']:>8}"
            f"{result['accuracy']:>10.4f}"
            f"{result['f1']:>10.4f}"
            f"{result['roc_auc']:>10.4f}"
            f"{result['pr_auc']:>10.4f}"
        )

    print(
        "-" * 85
    )

    # ========================================================================
    # SAVE SUMMARY
    # ========================================================================

    summary_path = os.path.join(
        OUTPUT_DIR,
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4
        )

    print()

    print(
        f"Summary saved:"
    )

    print(
        f"  {summary_path}"
    )

    print()

    print("=" * 70)

    print(
        "ZERO-SHOT EVALUATION COMPLETE"
    )

    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
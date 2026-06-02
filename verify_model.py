import ast
import warnings
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_samples():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = {getattr(target, "id", None) for target in node.targets}
            if "SAMPLES" in target_names:
                return ast.literal_eval(node.value)
    raise RuntimeError("Could not find SAMPLES in app.py")


def build_row(sample, scaler, feature_columns):
    scaled_amount, scaled_time = scaler.transform(
        [[sample["Amount"], sample["Time"]]]
    )[0]

    row = []
    for column in feature_columns:
        if column == "Time":
            row.append(scaled_time)
        elif column == "Amount":
            row.append(scaled_amount)
        else:
            row.append(sample.get(column, 0.0))
    return np.array(row).reshape(1, -1)


def main():
    model = joblib.load(ROOT / "fraud_detection_model.pkl")
    scaler = joblib.load(ROOT / "fraud_scaler.pkl")
    feature_columns = joblib.load(ROOT / "feature_columns.pkl")
    samples = load_samples()

    expected_predictions = {
        "normal": 0,
        "suspicious": 0,
        "fraud": 1,
    }

    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names.*",
        category=UserWarning,
    )

    print("Model verification")
    print(f"Model: {type(model).__name__}")
    print(f"Features: {len(feature_columns)}")
    print()

    failed = False
    for name, sample in samples.items():
        row = build_row(sample, scaler, feature_columns)
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])
        expected = expected_predictions[name]
        status = "PASS" if prediction == expected else "FAIL"
        failed = failed or status == "FAIL"

        print(
            f"{status:4} {name:10} "
            f"expected={expected} predicted={prediction} "
            f"fraud_probability={probability:.2%}"
        )

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

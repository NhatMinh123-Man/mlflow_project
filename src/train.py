import argparse
import json
import os
import time
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Đường dẫn
ROOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
MODEL_DIR = os.path.join(ROOT_DIR, "model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
BEST_ACC_PATH = os.path.join(MODEL_DIR, "accuracy.txt")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


def read_best_accuracy():
    if os.path.exists(BEST_ACC_PATH):
        try:
            return float(open(BEST_ACC_PATH, "r").read().strip())
        except:
            return None
    return None


def save_best(model, acc, args, run_id=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    # lưu model
    joblib.dump(model, BEST_MODEL_PATH)
    # lưu accuracy
    with open(BEST_ACC_PATH, "w") as f:
        f.write(str(acc))
    # lưu metadata
    meta = {
        "n_features": int(args.n_features),
        "n_samples": int(args.n_samples),
        "n_informative": int(args.n_informative),
        "n_redundant": int(args.n_redundant),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "n_estimators": int(args.n_estimators),
        "max_depth": None if args.max_depth is None else int(args.max_depth),
        "best_accuracy": float(acc),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mlflow_run_id": run_id,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main(args):
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("n_samples", args.n_samples)
        mlflow.log_param("n_features", args.n_features)
        mlflow.log_param("n_informative", args.n_informative)
        mlflow.log_param("n_redundant", args.n_redundant)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Tạo dữ liệu
        X, y = make_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=args.n_informative,
            n_redundant=args.n_redundant,
            random_state=args.random_state,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

        # Model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        # Đánh giá
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # Lưu model vào artifact của run
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(
            f"[RUN] n_estimators={args.n_estimators} max_depth={args.max_depth} | "
            f"samples={args.n_samples} features={args.n_features} -> acc={acc:.4f}"
        )

        # So sánh & cập nhật best
        prev_best = read_best_accuracy()
        if prev_best is None or acc > prev_best:
            save_best(model, acc, args, run.info.run_id)
            print(
                f"[BEST UPDATED] {acc:.4f} > {prev_best} → lưu best_model.pkl / accuracy.txt / meta.json"
            )
            # Đăng ký vào Registry (tùy môi trường)
            try:
                mlflow.register_model(
                    f"runs:/{run.info.run_id}/model", "BestClassifier"
                )
                print("[REGISTRY] Đã đăng ký model 'BestClassifier'")
            except Exception as e:
                print(f"[REGISTRY] Bỏ qua (local/no registry). Lý do: {e}")
        else:
            print(f"[KEEP] acc={acc:.4f} <= prev_best={prev_best:.4f} → giữ model cũ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hparams
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument(
        "--max_depth", type=lambda x: None if x == "None" else int(x), default=5
    )
    # Data params (mô phỏng làm giàu dữ liệu)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--n_informative", type=int, default=15)
    parser.add_argument("--n_redundant", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)

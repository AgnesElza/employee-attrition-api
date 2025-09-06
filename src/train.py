import argparse, os, json, time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

def train(data_path: str, target: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    if data_path.endswith(".feather"):
        df = pd.read_feather(data_path)
    else:
        df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {df.columns.tolist()}")

    # Map typical IBM HR Attrition target if it's Yes/No
    y = df[target]
    if y.dtype == object:
        y = y.map({"Yes":1, "No":0}).fillna(y)

    X = df.drop(columns=[target])

    # Auto-infer feature types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    pre = ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    clf = Pipeline(steps=[("pre", pre), ("model", LogisticRegression(max_iter=1000))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)

    model_path = os.path.join(outdir, "model.joblib")
    joblib.dump(clf, model_path)

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": data_path,
        "target": target,
        "auc": float(auc),
        "n_rows": int(len(df)),
        "n_features_before_encoding": int(X.shape[1])
    }
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save small reference sample for drift monitoring
    ref_out = "data/reference_sample.csv"
    os.makedirs("data", exist_ok=True)
    df.sample(min(1000, len(df)), random_state=42).to_csv(ref_out, index=False)

    print(f"Saved model to {model_path}")
    print(f"AUC: {auc:.4f}")
    print(f"Reference sample -> {ref_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV file with features + target column")
    p.add_argument("--target", default="Attrition", help="Target column name (default: Attrition)")
    p.add_argument("--outdir", default="models", help="Output directory for model artifacts")
    args = p.parse_args()
    train(args.data, args.target, args.outdir)

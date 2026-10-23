import time, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from features import normalize

df = pd.read_csv("landmarks.csv", header=None, dtype={0: str})
y = df[0].values
X = np.array([normalize(r) for r in df.iloc[:, 1:].values])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train: {len(X_tr)}, test: {len(X_te)}")

models = {
    "SVM": SVC(kernel="rbf", C=10),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300),
}

best_name, best_acc, best_model = None, 0, None
for name, m in models.items():
    t = time.time()
    m.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, m.predict(X_te))
    print(f"{name}: accuracy={acc:.4f}, час навчання={time.time()-t:.1f}с")
    if acc > best_acc:
        best_name, best_acc, best_model = name, acc, m

print("Найкраща:", best_name)
joblib.dump(best_model, "models/best.joblib")

fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay.from_estimator(best_model, X_te, y_te, ax=ax, colorbar=False)
plt.savefig("confusion_matrix.png", dpi=150)
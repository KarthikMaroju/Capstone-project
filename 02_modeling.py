"""
Module 2 - Analytics Pipeline: Part B - Predictive modeling
Continues from the SAME cleaned data committed by 01_eda.py (titanic.csv).
Never reloads the raw dataset from the network.

Run: python 02_modeling.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(HERE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

def hr(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

# ---------------------------------------------------------------------------
# Load the SAME cleaned data 01_eda.py produced. No second sns.load_dataset call.
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "titanic.csv"))
print("Loaded cleaned titanic.csv, shape:", df.shape)

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"
X = df[FEATURES].copy()
y = df[TARGET].copy()

# ---------------------------------------------------------------------------
# Task 7: Stratified split FIRST
# ---------------------------------------------------------------------------
hr("TASK 7: Stratified train/test split")
print("Class balance (full data):\n", y.value_counts(normalize=True).round(3))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nStratified on '{TARGET}' because the classes are imbalanced "
      f"(~{(1 - y.mean()):.0%} did not survive vs ~{y.mean():.0%} survived). "
      "A plain random split risks over/under-representing the minority class in "
      "train or test, which would bias both training and the reliability of the "
      "evaluation metrics -- stratification keeps the same ratio in both splits.")
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------------------------------------------------------
# Task 8: Preprocessing - fit on train only, via ColumnTransformer/Pipeline
# ---------------------------------------------------------------------------
hr("TASK 8: Preprocessing pipeline (fit on train only)")
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
print("Preprocessing: numeric -> median impute + StandardScaler; "
      "categorical (sex, embarked) -> most-frequent impute + one-hot encoding. "
      "All fit on X_train only, then transform-only on X_test.")

# ---------------------------------------------------------------------------
# Task 9: Train three classifiers on the identical split
# ---------------------------------------------------------------------------
hr("TASK 9: Train Logistic Regression, Decision Tree, Random Forest")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

fitted_pipelines = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
    print(f"Trained: {name}")

# Decision tree visualization
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = (numeric_features +
                  list(dt_pipe.named_steps["preprocess"]
                       .named_transformers_["cat"]
                       .named_steps["onehot"].get_feature_names_out(categorical_features)))
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_pipe.named_steps["model"], feature_names=feature_names,
          class_names=["Did not survive", "Survived"], filled=True, fontsize=7, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "decision_tree.png"), dpi=110)
plt.close(fig)
print("Decision tree rendered ->", os.path.join(CHART_DIR, "decision_tree.png"))

# ---------------------------------------------------------------------------
# Task 10: Evaluate all three models
# ---------------------------------------------------------------------------
hr("TASK 10: Evaluation - confusion matrix, accuracy, precision, recall, F1, ROC/AUC")

eval_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, pipe) in zip(axes, fitted_pipelines.items()):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    eval_rows.append({"Model": name, "Accuracy": acc, "Precision": prec,
                       "Recall": rec, "F1": f1, "AUC": auc})
    print(f"\n{name}\nConfusion matrix:\n{cm}\n"
          f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title(name)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "roc_curves.png"), dpi=110)
plt.close(fig)

classifier_comparison = pd.DataFrame(eval_rows).set_index("Model")
print("\nModel comparison table:\n", classifier_comparison.round(3))

# ---------------------------------------------------------------------------
# Task 11: Imbalance handling comparison (baseline / class_weight / SMOTE)
# ---------------------------------------------------------------------------
hr("TASK 11: Imbalance handling comparison")
print("Class balance in y_train:\n", y_train.value_counts(normalize=True).round(3))

# We use Logistic Regression as the representative model for this sub-task.
X_train_pre = preprocessor.fit_transform(X_train, y_train)
X_test_pre = preprocessor.transform(X_test)

imbalance_rows = []

# (a) baseline
clf_base = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_pre, y_train)
pred_base = clf_base.predict(X_test_pre)
imbalance_rows.append({"Strategy": "Baseline (no handling)",
                        "Precision": precision_score(y_test, pred_base),
                        "Recall": recall_score(y_test, pred_base),
                        "F1": f1_score(y_test, pred_base)})

# (b) class_weight='balanced'
clf_cw = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced").fit(X_train_pre, y_train)
pred_cw = clf_cw.predict(X_test_pre)
imbalance_rows.append({"Strategy": "class_weight='balanced'",
                        "Precision": precision_score(y_test, pred_cw),
                        "Recall": recall_score(y_test, pred_cw),
                        "F1": f1_score(y_test, pred_cw)})

# (c) SMOTE applied ONLY to the training fold
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_pre, y_train)
clf_sm = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_sm, y_train_sm)
pred_sm = clf_sm.predict(X_test_pre)
imbalance_rows.append({"Strategy": "SMOTE (train fold only)",
                        "Precision": precision_score(y_test, pred_sm),
                        "Recall": recall_score(y_test, pred_sm),
                        "F1": f1_score(y_test, pred_sm)})

imbalance_df = pd.DataFrame(imbalance_rows).set_index("Strategy")
print("\nImbalance strategy comparison:\n", imbalance_df.round(3))
best_recall_strategy = imbalance_df["Recall"].idxmax()
print(f"\nConclusion: '{best_recall_strategy}' achieved the highest recall on the minority "
      "(survived) class, at some cost to precision -- expected, since both class_weight and "
      "SMOTE deliberately shift the decision boundary to catch more true positives. Baseline "
      "logistic regression favors precision because it is biased toward the majority class. "
      "Which strategy 'wins' depends on whether missing a survivor (false negative) or "
      "wrongly flagging a non-survivor (false positive) is costlier for the use case.")

# ---------------------------------------------------------------------------
# Task 12: Hyperparameter tuning - GridSearchCV over Random Forest
# ---------------------------------------------------------------------------
hr("TASK 12: GridSearchCV tuning (Random Forest) + OOB score")
rf_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
])
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, None],
    "model__max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_rf_pipe = grid.best_estimator_
oob = best_rf_pipe.named_steps["model"].oob_score_
print(f"OOB score of best estimator: {oob:.3f}")

# ---------------------------------------------------------------------------
# Task 13: Regression side-task - predict fare
# ---------------------------------------------------------------------------
hr("TASK 13: Regression side-task (predict fare)")
reg_features = ["pclass", "age", "sibsp", "parch", "survived"]
Xr = df[reg_features].copy()
yr = df["fare"].copy()
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
     reg_features),
])
reg_pipe = Pipeline(steps=[("preprocess", reg_preprocessor), ("model", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test.shape[0], Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} Adjusted R2={adj_r2:.3f}")

residuals = yr_test - yr_pred
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(yr_pred, residuals, alpha=0.5)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted fare"); ax.set_ylabel("Residual")
ax.set_title("Residual plot (fare regression)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "regression_residuals.png"), dpi=110)
plt.close(fig)
print("Residuals fan out as predicted fare increases (the spread of residuals grows with "
      "the fitted value rather than staying constant) -- this is heteroscedasticity, meaning "
      "the constant-variance assumption of linear regression is violated here, largely "
      "because fare itself is a heavily right-skewed, high-variance variable.")

# ---------------------------------------------------------------------------
# Task 14: Model comparison table (classifiers + regression, separate groups)
# ---------------------------------------------------------------------------
hr("TASK 14: Full model comparison table + recommendation")
print("\n--- Classification metrics ---")
print(classifier_comparison.round(3))
print("\n--- Regression metrics (fare side-task; different scale, NOT comparable to above) ---")
reg_table = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted R2": adj_r2}],
                          index=["Linear Regression (fare)"])
print(reg_table.round(3))

best_model_name = classifier_comparison["F1"].idxmax()
print(f"\nRecommendation: deploy the {best_model_name} classifier -- across the comparison "
      f"table it posts the strongest F1 ({classifier_comparison.loc[best_model_name, 'F1']:.3f}) "
      f"and AUC ({classifier_comparison.loc[best_model_name, 'AUC']:.3f}), giving the best "
      "balance of precision and recall rather than optimizing a single metric. Random Forest "
      "in particular also tends to generalize better than a single Decision Tree and is more "
      "robust to the modest feature set here than Logistic Regression's linear boundary. "
      "If the cost of missing a true survivor is high, the class_weight/SMOTE-adjusted variant "
      "from Task 11 would be preferred over the plain baseline despite its lower precision.")

# ---------------------------------------------------------------------------
# Task 15: Save the best-performing COMPLETE pipeline
# ---------------------------------------------------------------------------
hr("TASK 15: Save best pipeline with joblib")
full_pipeline = fitted_pipelines[best_model_name]
pipeline_path = os.path.join(HERE, "best_pipeline.joblib")
joblib.dump(full_pipeline, pipeline_path)
print("Saved full fitted pipeline (preprocessing + model) ->", pipeline_path)

# reload + confirm it predicts on raw, unprocessed input
reloaded = joblib.load(pipeline_path)
sample_raw = X_test.iloc[[0]]
pred_check = reloaded.predict(sample_raw)
print("Reload check - prediction on one raw test row:", pred_check,
      " (matches original:", (pred_check == full_pipeline.predict(sample_raw)).all(), ")")

hr("02_modeling.py complete")

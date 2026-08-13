"""
Module 2 - Analytics Pipeline: Part A - Profiling, Cleaning, Data Story
Loads the Titanic dataset ONCE via seaborn (network/cache), profiles it,
cleans it per a percentage-based threshold rule, and produces the
required univariate / bivariate / multivariate analysis and charts.

Run:  python 01_eda.py
Produces: titanic.csv (offline fallback), charts/*.png, and printed output
that should be captured into the module README / notebook output.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(HERE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

def hr(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

# ---------------------------------------------------------------------------
# Task 1: Load ONCE, profile, save offline fallback CSV
# ---------------------------------------------------------------------------
hr("TASK 1: Load and profile")
df = sns.load_dataset("titanic")  # the ONE and ONLY network/cache load in this module
print("shape:", df.shape)
print("\ndf.info():")
df.info()
print("\ndf.describe():")
print(df.describe(include="all"))

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\nPercentage missing (columns with any missing values):")
print(missing_pct)

csv_path = os.path.join(HERE, "titanic.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved offline fallback -> {csv_path}")

# ---------------------------------------------------------------------------
# Task 2: Missing-value handling per column, threshold rule
#   <5% missing   -> drop those rows
#   5-30% missing -> impute
#   >30% missing  -> drop column OR encode "missing" (justify)
# ---------------------------------------------------------------------------
hr("TASK 2: Missing-value handling (threshold rule)")
df_clean = df.copy()

decisions = []
for col, pct in missing_pct.items():
    if pct < 5:
        before = len(df_clean)
        df_clean = df_clean[df_clean[col].notna()]
        decisions.append(
            f"- {col}: {pct}% missing (<5%) -> DROPPED the {before - len(df_clean)} affected rows."
        )
    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            fill_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill_val)
            decisions.append(
                f"- {col}: {pct}% missing (5-30%) -> IMPUTED with median ({fill_val:.2f})."
            )
        else:
            fill_val = df_clean[col].mode().iloc[0]
            df_clean[col] = df_clean[col].fillna(fill_val)
            decisions.append(
                f"- {col}: {pct}% missing (5-30%) -> IMPUTED with mode ('{fill_val}')."
            )
    else:
        # >30% missing: 'deck' is ~77% missing. Imputing would be unreliable
        # (we'd be inventing a deck for 3 out of 4 passengers), and it's not
        # required elsewhere in the pipeline, so we DROP the column rather
        # than encode "missing" as its own category.
        df_clean = df_clean.drop(columns=[col])
        decisions.append(
            f"- {col}: {pct}% missing (>30%) -> DROPPED THE COLUMN "
            f"(imputing >30% missing would be unreliable; column not needed downstream)."
        )

print("Decisions made:")
for d in decisions:
    print(d)

print("\nRemaining missing values after cleaning:\n", df_clean.isna().sum()[df_clean.isna().sum() > 0])
print("\nCleaned shape:", df_clean.shape)

# Re-save the CLEANED csv as titanic.csv is meant to be the working dataset
# for the rest of this module and for 02_modeling.py.
df_clean.to_csv(csv_path, index=False)
print(f"Re-saved cleaned dataset -> {csv_path}")

# ---------------------------------------------------------------------------
# Task 3: Univariate analysis - age & fare
# ---------------------------------------------------------------------------
hr("TASK 3: Univariate analysis (age, fare)")

def iqr_outliers(series, name):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lo) | (series > hi)]
    print(f"{name}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} bounds=[{lo:.2f}, {hi:.2f}] "
          f"-> {len(outliers)} outliers")
    return outliers

for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df_clean[col], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_title(f"{col} histogram")
    axes[1].boxplot(df_clean[col], vert=True)
    axes[1].set_title(f"{col} boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, f"univariate_{col}.png"), dpi=110)
    plt.close(fig)
    iqr_outliers(df_clean[col], col)

fare_mean, fare_median, fare_mode = df_clean["fare"].mean(), df_clean["fare"].median(), df_clean["fare"].mode().iloc[0]
print(f"\nfare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
skew_dir = "right-skewed" if fare_mean > fare_median > fare_mode else (
    "left-skewed" if fare_mean < fare_median < fare_mode else "roughly symmetric")
print(f"fare distribution is {skew_dir}: mean > median > mode indicates a long right tail "
      f"driven by a small number of high-fare (first-class) passengers.")

# ---------------------------------------------------------------------------
# Task 4: Bivariate analysis - boolean masking + correlation matrix
# ---------------------------------------------------------------------------
hr("TASK 4: Bivariate analysis")

surv_by_sex = df_clean[df_clean["sex"] == "female"]["survived"].mean(), df_clean[df_clean["sex"] == "male"]["survived"].mean()
print(f"Survival rate by sex -> female: {surv_by_sex[0]:.3f}, male: {surv_by_sex[1]:.3f}")

print("\nSurvival rate by pclass:")
for p in sorted(df_clean["pclass"].unique()):
    rate = df_clean[df_clean["pclass"] == p]["survived"].mean()
    print(f"  pclass {p}: {rate:.3f}")

print("\nSurvival rate by sex & pclass:")
for s in ["female", "male"]:
    for p in sorted(df_clean["pclass"].unique()):
        mask = (df_clean["sex"] == s) & (df_clean["pclass"] == p)
        rate = df_clean.loc[mask, "survived"].mean()
        print(f"  sex={s}, pclass={p}: {rate:.3f}  (n={mask.sum()})")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df_clean[corr_cols].corr()
print("\nCorrelation matrix (6x6):\n", corr.round(3))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation heatmap (survived, pclass, age, sibsp, parch, fare)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "correlation_heatmap.png"), dpi=110)
plt.close(fig)

# two strongest off-diagonal correlations by absolute value
pairs = []
for i, c1 in enumerate(corr_cols):
    for c2 in corr_cols[i + 1:]:
        pairs.append((c1, c2, corr.loc[c1, c2]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nTop 2 strongest correlations (by |r|):")
for c1, c2, r in pairs[:2]:
    print(f"  {c1} <-> {c2}: r = {r:.3f}")
c1, c2, r = pairs[0]
c1b, c2b, rb = pairs[1]
print(f"Interpretation: the strongest relationship is {c1}<->{c2} (r={r:.3f}) -- higher-class "
      f"(numerically lower pclass) tickets cost more, so the correlation with fare is "
      f"strongly negative. The second strongest is {c1b}<->{c2b} (r={rb:.3f}) -- passengers "
      f"traveling with more siblings/spouses also tended to travel with more parents/children, "
      f"i.e. family groups moved together, which is a family-size signal rather than a "
      f"survival driver on its own.")

# ---------------------------------------------------------------------------
# Task 5: Multivariate "data story" - at least 4 charts
# ---------------------------------------------------------------------------
hr("TASK 5: Multivariate data story (>=4 charts)")

# Chart 1: survival rate by class and sex (bar)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", ax=ax, errorbar=None)
ax.set_title("Survival rate by class and sex")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "story_1_survival_by_class_sex.png"), dpi=110)
plt.close(fig)
print("Chart 1 (bar): Survival rate by class and sex -> Women in 1st and 2nd class survived "
      "at very high rates (>90%), while men in 2nd and 3rd class survived at the lowest "
      "rates. Sex is a stronger predictor of survival than class alone, but the two "
      "compound: being both male and 3rd class was the worst combination.")

# Chart 2: age distribution by survival (box)
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df_clean, x="survived", y="age", ax=ax)
ax.set_title("Age distribution by survival outcome")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "story_2_age_by_survival.png"), dpi=110)
plt.close(fig)
print("Chart 2 (box): Age by survival -> Median age is similar between survivors and "
      "non-survivors, but survivors include more very young children (consistent with "
      "'women and children first'), while the non-survivor group has a slightly heavier "
      "concentration of working-age men.")

# Chart 3: fare vs age scatter, colored by survival
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", alpha=0.6, ax=ax)
ax.set_title("Fare vs age, colored by survival")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "story_3_fare_age_scatter.png"), dpi=110)
plt.close(fig)
print("Chart 3 (scatter): Fare vs age by survival -> Survivors cluster at higher fares "
      "across all ages, reinforcing that ticket price (a proxy for class and deck location) "
      "mattered more to survival odds than age by itself.")

# Chart 4: pair plot of the numeric story variables
pp = sns.pairplot(df_clean[["survived", "pclass", "age", "fare"]], hue="survived",
                   diag_kind="hist", plot_kws={"alpha": 0.5})
pp.fig.suptitle("Pairwise relationships: survived, pclass, age, fare", y=1.02)
pp.savefig(os.path.join(CHART_DIR, "story_4_pairplot.png"), dpi=110)
plt.close(pp.fig)
print("Chart 4 (pair plot): Across every pairing, pclass and fare separate survivors from "
      "non-survivors more cleanly than age does -- together the four charts build the "
      "argument that survival was driven primarily by class/fare (proxy for deck and "
      "lifeboat access) and sex, with age playing a secondary, mostly child-priority role.")

# ---------------------------------------------------------------------------
# Task 6: EDA-stage standardization check (does NOT feed into modeling)
# ---------------------------------------------------------------------------
hr("TASK 6: EDA-stage standardization check (age, fare)")
before = df_clean[["age", "fare"]].agg(["mean", "std"])
print("Before standardization:\n", before)

z = (df_clean[["age", "fare"]] - df_clean[["age", "fare"]].mean()) / df_clean[["age", "fare"]].std()
after = z.agg(["mean", "std"])
print("\nAfter z-score standardization (approx mean 0, std 1):\n", after.round(6))
print("\nNote: this is an EDA-stage sanity check only. The modeling pipeline in "
      "02_modeling.py fits its own StandardScaler on the TRAIN split only.")

hr("01_eda.py complete")
print("Charts saved to:", CHART_DIR)

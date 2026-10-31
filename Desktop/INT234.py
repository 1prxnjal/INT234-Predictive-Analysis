import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             silhouette_score)

''' plt.rcParams['figure.dpi']        = 150   
plt.rcParams['savefig.dpi']       = 300    
plt.rcParams['figure.figsize']    = (9, 5) 
plt.rcParams['font.size']         = 11
plt.rcParams['axes.titlesize']    = 13
plt.rcParams['axes.labelsize']    = 11
plt.rcParams['axes.grid']         = True
plt.rcParams['grid.alpha']        = 0.3 '''
sns.set_style("whitegrid")


df = pd.read_csv("D:\LPU\LPU-Sem5\INT234\iitm_subdivision_rainfall_1901_2016.csv")
months = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]


print(df.info())
print("\nDescription:\n", df[months].describe().round(1))
print("\nMissing values:\n", df.isna().sum().sum())
print("\nDuplicate (Year, Subdivision):",
      df.duplicated(subset=["Year","Subdivision"]).sum())
print("\nNegative rainfall values:",
      (df[months] < 0).sum().sum())
print("\nNumber of subdivisions:", df["Subdivision"].nunique())
print("\nYear range:", df["Year"].min(), "-", df["Year"].max())


df["JF"]   = df[["January","February"]].sum(axis=1)
df["MAM"]  = df[["March","April","May"]].sum(axis=1)
df["JJAS"] = df[["June","July","August","September"]].sum(axis=1)
df["OND"]  = df[["October","November","December"]].sum(axis=1)
df["ANN"]  = df[months].sum(axis=1)

#EDA
#Distribution of annual rainfall
plt.figure(figsize=(9,5))
sns.histplot(df["ANN"], bins=50, kde=True, color="steelblue")
plt.title("Distribution of Annual Rainfall (1901–2016)")
plt.xlabel("Annual Rainfall (mm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("fig1_annual_distribution.png", bbox_inches="tight")
plt.show()

#Monthly climatology heatmap (subdivision × month)
clim = df.groupby("Subdivision")[months].mean()
clim = clim.loc[clim.mean(axis=1).sort_values(ascending=False).index]

plt.figure(figsize=(12, 9))
sns.heatmap(clim, cmap="YlGnBu", cbar_kws={"label": "Mean monthly rainfall (mm)"})
plt.title("Mean Monthly Rainfall per Subdivision")
plt.xlabel("Month")
plt.ylabel("Subdivision")
plt.tight_layout()
plt.savefig("fig2_climatology_heatmap.png", bbox_inches="tight")
plt.show()

#Monsoon spread by subdivision
order = df.groupby("Subdivision")["JJAS"].median().sort_values(ascending=False).index

plt.figure(figsize=(13, 5))
sns.boxplot(data=df, x="Subdivision", y="JJAS", order=order, palette="viridis")
plt.xticks(rotation=90)
plt.title("Monsoon (JJAS) Rainfall per Subdivision")
plt.ylabel("JJAS Rainfall (mm)")
plt.tight_layout()
plt.savefig("fig3_jjas_boxplot.png", bbox_inches="tight")
plt.show()

#Correlation between months
plt.figure(figsize=(10, 8))
sns.heatmap(df[months].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0)
plt.title("Monthly Rainfall Correlation")
plt.tight_layout()
plt.savefig("fig4_correlation.png", bbox_inches="tight")
plt.show()

#Time series of monsoon for contrasting subdivisions
plt.figure(figsize=(12, 5))
for sub in ["KERALA", "WEST RAJASTHAN", "ASSAM & MEGHALAYA", "TAMIL NADU"]:
    s = df[df["Subdivision"] == sub].sort_values("Year")
    if not s.empty:
        plt.plot(s["Year"], s["JJAS"], label=sub, alpha=0.9, linewidth=1.6)
plt.legend()
plt.title("Monsoon (JJAS) Rainfall Time Series")
plt.xlabel("Year")
plt.ylabel("JJAS Rainfall (mm)")
plt.tight_layout()
plt.savefig("fig5_time_series.png", bbox_inches="tight")
plt.show()
TARGET = "JJAS"
dummies = pd.get_dummies(df["Subdivision"], prefix="sd", drop_first=True).astype(int)

base = pd.concat([
    df[["Year","JF","MAM",TARGET]].reset_index(drop=True),
    dummies.reset_index(drop=True)
], axis=1)
base["Subdivision"] = df["Subdivision"].values 

feature_cols = ["Year","JF","MAM"] + list(dummies.columns)
X_reg = base[feature_cols]
y_reg = base[TARGET]


base["high_monsoon"] = (
    base[TARGET] > base.groupby("Subdivision")[TARGET].transform("median")
).astype(int)
y_cls = base["high_monsoon"]

# Train / test splits 
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)
Xc_train, Xc_test, y_cls_train, y_cls_test = train_test_split(
    X_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

# MinMax normalization 
scaler = MinMaxScaler()
Xc_train_scaled = scaler.fit_transform(Xc_train)
Xc_test_scaled  = scaler.transform(Xc_test)

print("\nClass balance (high_monsoon):",
      y_cls.value_counts(normalize=True).round(3).to_dict())

#Model 1 : MLR
lin_model = LinearRegression()
lin_model.fit(X_train, y_reg_train)
y_pred_lin = lin_model.predict(X_test)

lin_metrics = {
    "MAE":  mean_absolute_error(y_reg_test, y_pred_lin),
    "MSE":  mean_squared_error(y_reg_test, y_pred_lin),
    "RMSE": np.sqrt(mean_squared_error(y_reg_test, y_pred_lin)),
    "R2":   r2_score(y_reg_test, y_pred_lin)
}
print("\n=== Multiple Linear Regression ===")
for k, v in lin_metrics.items():
    print(f"{k}: {v:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(7, 6))
plt.scatter(y_reg_test, y_pred_lin, alpha=0.5, edgecolors="k", s=25)
lims = [min(y_reg_test.min(), y_pred_lin.min()),
        max(y_reg_test.max(), y_pred_lin.max())]
plt.plot(lims, lims, "r--", linewidth=2, label="Perfect fit")
plt.xlabel("Actual JJAS (mm)")
plt.ylabel("Predicted JJAS (mm)")
plt.title("MLR — Actual vs Predicted (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("fig6_mlr_actual_vs_pred.png", bbox_inches="tight")
plt.show()

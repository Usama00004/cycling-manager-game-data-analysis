# ===============================
# Import necessary libraries
# ===============================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import requests
import re

# ===============================
# Load Dataset
# ===============================
def load_data(url: str) -> pd.DataFrame:
    """Load and properly parse the cycling data manually from the URL."""
    try:
        # Fetch raw text data
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.strip().split("\n")

        data = []
        for line in lines:
            # Split by whitespace, but keep quoted strings together
            parts = line.replace('"', '').split()
            # The data follows the pattern: rider, rider_class, stage, points, stage_class
            if len(parts) >= 5:
                rider = parts[0] + " " + parts[1]  # combine first and last name
                rider_class = parts[2] + " " + parts[3] if parts[3] not in ["X1", "X2", "X3"] else parts[2]
                stage = parts[-3]
                points = parts[-2]
                stage_class = parts[-1]
                data.append([rider, rider_class, stage, points, stage_class])

        df = pd.DataFrame(data, columns=["all_riders", "rider_class", "stage", "points", "stage_class"])
        print("Data loaded successfully!")
        print(df.head())
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


# ===============================
# Clean & Prepare Data
# ===============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by ensuring correct types and removing missing values."""
    if df.empty:
        raise ValueError("DataFrame is empty. Check loading function.")
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df.dropna(subset=["points", "rider_class", "stage_class"], inplace=True)
    print("Data cleaned and formatted.")
    return df


# ===============================
# Descriptive Statistics
# ===============================
def descriptive_statistics(df: pd.DataFrame):
    """Generate descriptive statistics grouped by rider and stage class."""
    desc = df.groupby(["rider_class", "stage_class"])["points"].describe()
    print("\n Descriptive Statistics by Rider and Stage Class:")
    print(desc)
    return desc


# ===============================
# Visual Analysis
# ===============================
def plot_rider_performance(df: pd.DataFrame):
    """Plot average rider performance across stage types."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x="stage_class", y="points", hue="rider_class", data=df, ci="sd", palette="muted")
    plt.title("Average Rider Performance by Stage Class")
    plt.xlabel("Stage Class")
    plt.ylabel("Average Points")
    plt.legend(title="Rider Class")
    plt.tight_layout()
    plt.show()


def boxplot_rider_points(df: pd.DataFrame):
    """Boxplot showing distribution of points by rider class."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="rider_class", y="points", data=df, palette="pastel")
    plt.title("Distribution of Points per Rider Class")
    plt.xlabel("Rider Class")
    plt.ylabel("Points")
    plt.tight_layout()
    plt.show()


def anova_test(df: pd.DataFrame):
    print("\n One-way ANOVA Test:")
    groups = [df.loc[df["rider_class"] == c, "points"] for c in df["rider_class"].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"F-statistic = {f_stat:.3f}, p-value = {p_value:.3e}")
    return f_stat, p_value

def plot_violin(df, x_col='stage_class', y_col='points', hue_col='rider_class_clean'):
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x=x_col,
        y=y_col,
        hue=hue_col,
        data=df,
        inner='quartile',
        palette='Set2',
        split=True
    )
    plt.title('Distribution of Rider Points by Stage Class', fontsize=14)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    plt.legend(title=hue_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def stage_class_summary(df):
    valid_classes = ["Flat", "Hills", "mount"]
    df["stage_class_clean"] = df["stage_class"].apply(
        lambda x: next((c for c in valid_classes if c.lower() in str(x).lower()), None)
    )
    df = df.dropna(subset=["stage_class_clean"])
    summary = df.groupby("stage_class_clean")["points"].agg(
        Mean="mean",
        Median="median",
        Std_Dev="std",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    ).reset_index()
    summary["IQR"] = summary["Q3"] - summary["Q1"]
    summary = summary.drop(columns=["Q1", "Q3"])
    summary = summary.round(1)
    summary = summary.rename(columns={"stage_class_clean": "Stage Class"})
    return summary

def rider_class_summary(df):
    df = clean_rider_class(df)
    summary = df.groupby("rider_class_clean")["points"].agg(
        Mean="mean",
        Median="median",
        Std_Dev="std",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    ).reset_index()

    summary["IQR"] = summary["Q3"] - summary["Q1"]
    summary = summary.drop(columns=["Q1", "Q3"])
    summary = summary.round(1)
    summary = summary.rename(columns={"rider_class_clean": "Rider Class"})
    return summary


def clean_rider_class(df):
    valid_classes = ["All Rounder", "Climber", "Sprinter", "Unclassed"]
    df["rider_class_clean"] = df["rider_class"].apply(
        lambda x: next((c for c in valid_classes if c in x), None)
    )
    df = df.dropna(subset=["rider_class_clean"])
    return df

# ===============================
# Main Function
# ===============================
def main():
    url = "https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/Studium/Studiengaenge-Infos/Data_Science/cycling.txt"
    df = load_data(url)
    df = clean_data(df)
    df = clean_rider_class(df)
    plot_violin(df)

# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    main()

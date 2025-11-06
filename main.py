import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import requests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# Load Dataset
def load_data(url: str) -> pd.DataFrame:
    """Load and properly parse the cycling data manually from the URL."""
    try:
        # Fetch raw text data
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.strip().split("\n")

        data = []
        for line in lines:
            parts = line.replace('"', '').split()
            if len(parts) >= 5:
                rider = parts[0] + " " + parts[1]
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


# Clean & Prepare Data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by ensuring correct types and removing missing values."""
    if df.empty:
        raise ValueError("DataFrame is empty. Check loading function.")
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df.dropna(subset=["points", "rider_class", "stage_class"], inplace=True)
    print("Data cleaned and formatted.")
    return df

def clean_rider_class(df):
    valid_classes = ["All Rounder", "Climber", "Sprinter", "Unclassed"]
    df["rider_class_clean"] = df["rider_class"].apply(
        lambda x: next((c for c in valid_classes if c in x), None)
    )
    df = df.dropna(subset=["rider_class_clean"])
    return df


# Descriptive Statistics

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

# Visual Analysis

def histogram_plot_points_by_rider_and_stage(df):
    grouped = df.groupby(['rider_class_clean', 'stage_class'])['points'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=grouped,
        x='rider_class_clean',
        y='points',
        hue='stage_class',
        # palette='gray',  # Black & white mode
        edgecolor='black'
    )

    plt.title('Total Points per Rider Class by Stage Class', fontsize=14)
    plt.xlabel('Rider Class', fontsize=12)
    plt.ylabel('Total Points', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Stage Class', frameon=True)
    plt.tight_layout()
    plt.show()

def boxplot_rider_points(df: pd.DataFrame):
    df = clean_rider_class(df)
    """Boxplot showing distribution of points by rider class."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="rider_class_clean", y="points", data=df, palette="pastel")
    plt.title("Distribution of Points per Rider Class")
    plt.xlabel("Rider Class")
    plt.ylabel("Points")
    plt.tight_layout()
    plt.show()


# Testing

def anova_with_tukey(df, dependent_var='points', factor_var='rider_class_clean', alpha=0.05):

    model = ols(f'{dependent_var} ~ C({factor_var})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2) 

    p_value = anova_table['PR(>F)'][0]
    significant = p_value < alpha

    tukey_summary = None
    if significant:
        tukey = pairwise_tukeyhsd(endog=df[dependent_var], groups=df[factor_var], alpha=alpha)
        tukey_summary = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    return {
        'anova_table': anova_table,
        'tukey_summary': tukey_summary,
        'significant': significant
    }

# Main Function

def main():
    url = "https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/Studium/Studiengaenge-Infos/Data_Science/cycling.txt"
    df = load_data(url)
    df = clean_data(df)
    df = clean_rider_class(df)
    print(df)



# Entry Point

if __name__ == "__main__":
    main()

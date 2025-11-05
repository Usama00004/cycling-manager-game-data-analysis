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
        print("✅ Data loaded successfully!")
        print(df.head())
        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()


# ===============================
# Clean & Prepare Data
# ===============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by ensuring correct types and removing missing values."""
    if df.empty:
        raise ValueError("❌ DataFrame is empty. Check loading function.")

    # Strip unwanted characters
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert points to numeric
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    # Drop rows with missing critical values
    df.dropna(subset=["points", "rider_class", "stage_class"], inplace=True)

    print("✅ Data cleaned and formatted.")
    return df


# ===============================
# 3. Descriptive Statistics
# ===============================
def descriptive_statistics(df: pd.DataFrame):
    """Generate descriptive statistics grouped by rider and stage class."""
    desc = df.groupby(["rider_class", "stage_class"])["points"].describe()
    print("\n📊 Descriptive Statistics by Rider and Stage Class:")
    print(desc)
    return desc


# ===============================
# 4. Visual Analysis
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


# ===============================
# 5. Hypothesis Testing
# ===============================
def test_normality(df: pd.DataFrame):
    """Perform Shapiro-Wilk test for normality."""
    print("\n🧪 Normality Test (Shapiro-Wilk):")
    results = {}
    for rider in df["rider_class"].unique():
        stat, p = stats.shapiro(df.loc[df["rider_class"] == rider, "points"])
        results[rider] = (stat, p)
        print(f"{rider}: statistic={stat:.3f}, p-value={p:.3f}")
    return results


def anova_test(df: pd.DataFrame):
    """Perform one-way ANOVA across rider classes."""
    print("\n📈 One-way ANOVA Test:")
    groups = [df.loc[df["rider_class"] == c, "points"] for c in df["rider_class"].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"F-statistic = {f_stat:.3f}, p-value = {p_value:.3e}")
    return f_stat, p_value


def pairwise_ttests(df: pd.DataFrame):
    """Conduct pairwise t-tests between rider classes."""
    from itertools import combinations
    classes = df["rider_class"].unique()
    print("\n🔍 Pairwise T-Tests between Rider Classes:")
    for c1, c2 in combinations(classes, 2):
        t_stat, p_value = stats.ttest_ind(
            df.loc[df["rider_class"] == c1, "points"],
            df.loc[df["rider_class"] == c2, "points"],
            equal_var=False
        )
        print(f"{c1} vs {c2}: t={t_stat:.3f}, p={p_value:.3f}")


def plot_violin(df, x_col='stage_class', y_col='points', hue_col='rider_class_clean'):
    """
    Creates a violin plot to visualize the distribution of points
    across stage classes, optionally grouped by rider class.

    Parameters:
        df (DataFrame): The input dataframe.
        x_col (str): Column name for the X-axis (default: 'stage_class').
        y_col (str): Column name for the Y-axis (default: 'points').
        hue_col (str): Column name for color grouping (default: 'rider_class_clean').
    """
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
    
    # Aggregate statistics
    summary = df.groupby("stage_class_clean")["points"].agg(
        Mean="mean",
        Median="median",
        Std_Dev="std",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    ).reset_index()
    
    # Calculate IQR
    summary["IQR"] = summary["Q3"] - summary["Q1"]
    
    # Drop Q1 and Q3 columns
    summary = summary.drop(columns=["Q1", "Q3"])
    
    # Round values for readability
    summary = summary.round(1)
    
    # Rename column
    summary = summary.rename(columns={"stage_class_clean": "Stage Class"})
    
    return summary




def rider_class_summary(df):
    """
    Calculates mean, median, standard deviation, and IQR of points
    grouped by cleaned rider_class.
    """
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
    """
    Normalizes rider_class to only the 4 valid classes.
    """
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
    
    # Load and clean data
    df = load_data(url)
    
    
    df = clean_data(df)
    

# ✅ Example usage:
# First, clean your data
    df = clean_rider_class(df)

# Then, create the violin plot
    plot_violin(df)
    
    # # Visualization
    #  plot_rider_performance(df)
     #  boxplot_rider_points(df)
    
    # # Hypothesis Testing
    # test_normality(df)
    # anova_test(df)
    # pairwise_ttests(df)

    # print("\n✅ Analysis complete!")


# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    main()

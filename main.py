# import pandas as pd

# def load_cycling_data():
#     """Loads the cycling dataset into a pandas DataFrame."""
#     url = "https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/Studium/Studiengaenge-Infos/Data_Science/cycling.txt"
    
#     df = pd.read_csv(url, sep="\t")
#     return df

# def main():
#     """Main function to execute the data loading process."""
#     df = load_cycling_data()
#     print("✅ Dataset loaded successfully!\n")
#     print(df.head())  

# if __name__ == "__main__":
#     main()



# cycling_analysis.py

# ===============================
# Import necessary libraries
# ===============================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import requests

# ===============================
# 1. Load Dataset
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
# 2. Clean & Prepare Data
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


# ===============================
# 6. Main Function
# ===============================
def main():
    url = "https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/Studium/Studiengaenge-Infos/Data_Science/cycling.txt"
    
    # Load and clean data
    df = load_data(url)
    
    
    df = clean_data(df)
 
    
    # # Descriptive Analysis
    descriptive_statistics(df)

    
    # # Visualization
    plot_rider_performance(df)
    boxplot_rider_points(df)
    
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

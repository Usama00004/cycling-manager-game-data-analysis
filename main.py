import pandas as pd

def load_cycling_data():
    """Loads the cycling dataset into a pandas DataFrame."""
    url = "https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/Studium/Studiengaenge-Infos/Data_Science/cycling.txt"
    
    # Read the dataset (tab-separated)
    df = pd.read_csv(url, sep="\t")
    return df

def main():
    """Main function to execute the data loading process."""
    df = load_cycling_data()
    print("✅ Dataset loaded successfully!\n")
    print(df.head())  # Show first 5 rows

if __name__ == "__main__":
    main()

#  Cycling Manager Game Data Analysis  
**Application Report for M.Sc. Data Science – TU Dortmund University (Summer Semester 2026)**  


## Project Overview
This project is part of the application requirement for the **Master’s program in Data Science at TU Dortmund University**.  
The dataset represents results from a **cycling manager game**, where professional riders earn points based on their performance in multiple race stages.

<img src="https://github.com/Usama00004/cycling-manager-game-data-analysis/blob/main/Image.png" alt="Distribution of total points by rider class and stage class" width="800" height="400">



The main objective of this analysis is to:
- Understand whether there is a **significant performance difference among rider classes**.
- Compare how different **rider types perform across various stage types** (flat, hills, mountain).


## Research Questions

1. Is there a difference in the average performance (points) between the rider classes?
2. How do rider classes perform on different stage types (flat, hills, mountain)?
3. Which statistical measures and tests can best describe and evaluate these differences?


## Dataset Description

- **Source:** [cycling.txt](https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/Studium/Studiengaenge-Infos/Data_Science/cycling.txt)  
- **Number of Observations:** ~3,500  
- **Variables:**
  | Variable | Description |
  |-----------|-------------|
  | `all_riders` | Name of the rider |
  | `rider_class` | Category of the rider (All Rounder, Climber, Sprinter, Unclassed) |
  | `stage` | Stage identifier (X1–X21) |
  | `points` | Points earned by the rider for that stage |
  | `stage_class` | Type of stage (flat, hills, mountain) |


## Methods and Analysis

The project is divided into two major analytical components:

### **1. Descriptive Analysis**
- Computation of descriptive statistics (mean, median, std, min, max) grouped by:
  - Rider class
  - Stage class
- Visualization of rider performance using:
  - **Box Plot** – to compare point distributions across rider classes.
  - **Bar Chart** – to show mean points across stage types.

### **2. Inferential Analysis (Hypothesis Testing)**
- Testing for statistical differences between groups using:
  - **ANOVA (Analysis of Variance)** – to compare means among rider classes.
  - **Post-hoc Tests (Tukey HSD)** – to identify specific group differences.



## Technologies Used

| Tool / Library | Purpose |
|-----------------|----------|
| **Python 3.11** | Programming language |
| **Pandas** | Data cleaning and manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Advanced statistical plotting |
| **SciPy** | Statistical tests (ANOVA, normality checks) |
| **Statsmodels** | Advanced hypothesis testing (Tukey HSD) |



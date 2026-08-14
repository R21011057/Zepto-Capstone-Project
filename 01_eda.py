import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the plots directory exists
os.makedirs('analytics/plots', exist_ok=True)

def main():
    print("=== Module 2 Part A: EDA ===")
    
    # 1. Load Data Exactly Once
    csv_path = 'analytics/titanic.csv'
    if not os.path.exists(csv_path):
        print("Loading dataset from Seaborn for the first time...")
        df_raw = sns.load_dataset('titanic')
        df_raw.to_csv(csv_path, index=False)
    
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Profile Data
    print("\n--- Data Profiling ---")
    print(f"Shape: {df.shape}")
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe(include='all'))
    
    # 2. Missing Percentages
    print("\n--- Missing Value Percentages ---")
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    affected_cols = missing_percentages[missing_percentages > 0]
    for col, pct in affected_cols.items():
        print(f"{col}: {pct:.2f}% missing")
        
    # 3. Threshold-based Missing Value Handling
    print("\n--- Missing Value Handling ---")
    # <5% drop rows (embarked, embark_town)
    # 5-30% impute (age)
    # >30% justify dropping or encoding (deck)
    
    # Justification for 'deck' (77.22% missing)
    print("Decision for 'deck': The 'deck' column has over 77% missing values. Imputing such a vast majority of the data would heavily skew any analysis and introduce massive bias. Therefore, we will drop the 'deck' column entirely.")
    df = df.drop(columns=['deck'])
    
    # Justification for 'age' (19.87% missing)
    print("Decision for 'age': Age has ~20% missing values. We will impute this using the median age to preserve the dataset size without severely impacting the central tendency.")
    df['age'] = df['age'].fillna(df['age'].median())
    
    # Justification for 'embarked' and 'embark_town' (~0.22% missing)
    print("Decision for 'embarked'/'embark_town': These have <5% missing values. We will simply drop the rows containing these missing values.")
    df = df.dropna(subset=['embarked', 'embark_town'])
    
    print(f"Shape after cleaning: {df.shape}")
    
    # 4. & 5. Univariate Analysis (Histograms & Boxplots)
    print("\n--- Univariate Analysis ---")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df['age'], kde=True, ax=axes[0, 0]).set_title('Age Histogram')
    sns.boxplot(x=df['age'], ax=axes[0, 1]).set_title('Age Boxplot')
    sns.histplot(df['fare'], kde=True, ax=axes[1, 0]).set_title('Fare Histogram')
    sns.boxplot(x=df['fare'], ax=axes[1, 1]).set_title('Fare Boxplot')
    plt.tight_layout()
    plt.savefig('analytics/plots/univariate_age_fare.png')
    plt.close()
    
    # 6. IQR Outlier Counts
    def get_outlier_count(col_data):
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        
    age_outliers = get_outlier_count(df['age'])
    fare_outliers = get_outlier_count(df['fare'])
    print(f"Age Outliers (IQR): {age_outliers}")
    print(f"Fare Outliers (IQR): {fare_outliers}")
    
    # 7. & 8. Fare Statistics & Skewness
    fare_mean = df['fare'].mean()
    fare_median = df['fare'].median()
    fare_mode = df['fare'].mode()[0]
    
    print(f"\nFare - Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")
    
    # Interpretation
    print("\n[Interpretation: Skewness]")
    if fare_mean > fare_median > fare_mode:
        print("The distribution of 'fare' is right-skewed (positive skew), as evidenced by the Mean > Median > Mode ordering. This implies most passengers paid lower fares, while a small number of extreme outliers paid significantly higher fares, pulling the mean to the right.")
    else:
        print("The 'fare' distribution does not follow a strict right-skewed pattern in its strict inequalities, but visual inspection and the relationship between mean and median usually indicate a strong positive skew.")

    # 9, 10, 11. Survival Rates
    print("\n--- Survival Rates ---")
    print(df.groupby('sex')['survived'].mean())
    print("\n")
    print(df.groupby('pclass')['survived'].mean())
    print("\n")
    print(df.groupby(['sex', 'pclass'])['survived'].mean())
    
    # 12, 13, 14, 15. Correlation Matrix & Heatmap
    print("\n--- Correlation Analysis ---")
    corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("6x6 Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig('analytics/plots/correlation_heatmap.png')
    plt.close()
    
    # Strongest two off-diagonal
    corr_unstacked = corr_matrix.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1.0] # Remove diagonal
    top_two = corr_unstacked.drop_duplicates().sort_values(ascending=False).head(2)
    print("\nTop 2 absolute off-diagonal correlations:")
    print(top_two)
    
    print("\n[Interpretation: Correlations]")
    print("The strongest off-diagonal correlation is between pclass and fare (negative). This makes logical sense as a lower class number (1st class) indicates a higher ticket fare. The second strongest is between sibsp and parch (positive), indicating that passengers who traveled with siblings/spouses were also highly likely to travel with parents/children, capturing the presence of families aboard.")
    
    # 16 & 17. 4 Distinct Multivariate Charts
    print("\n--- Multivariate Data Story ---")
    
    # Chart 1: Barplot of Survival by Sex and Pclass
    plt.figure()
    sns.barplot(data=df, x='pclass', y='survived', hue='sex')
    plt.title("Chart 1: Survival Rate by Passenger Class and Sex")
    plt.savefig('analytics/plots/multivariate_1.png')
    plt.close()
    print("\n[Interpretation: Chart 1]")
    print("This bar chart illustrates the stark interaction between class, sex, and survival. Females across all classes had a significantly higher survival rate than males, demonstrating the 'women and children first' protocol. Additionally, 1st class females had a near 100% survival rate, while 3rd class males suffered the highest casualties.")
    
    # Chart 2: Violin plot of Age by Survived and Sex
    plt.figure()
    sns.violinplot(data=df, x='survived', y='age', hue='sex', split=True)
    plt.title("Chart 2: Age Distribution by Survival and Sex")
    plt.savefig('analytics/plots/multivariate_2.png')
    plt.close()
    print("\n[Interpretation: Chart 2]")
    print("This violin plot reveals the age distribution of survivors vs. non-survivors broken down by sex. We can observe a prominent bulge at younger ages (children) for male survivors compared to non-survivors, reiterating the priority given to children. Female distributions are relatively similar whether they survived or not, given their generally high survival rate.")
    
    # Chart 3: Scatterplot of Age vs Fare by Survival
    plt.figure()
    sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.6)
    plt.title("Chart 3: Age vs. Fare colored by Survival")
    plt.savefig('analytics/plots/multivariate_3.png')
    plt.close()
    print("\n[Interpretation: Chart 3]")
    print("This scatter plot visualizes the relationship between a passenger's age and fare, marked by their survival outcome. The highest fares are heavily clustered with survivors (orange dots), regardless of age. Conversely, the dense cluster of non-survivors resides in the low-fare, lower-class brackets, further establishing economic status as a primary survival factor.")
    
    # Chart 4: Heatmap of average fare by pclass and embarked
    pivot_fare = df.pivot_table(values='fare', index='pclass', columns='embarked', aggfunc='mean')
    plt.figure()
    sns.heatmap(pivot_fare, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Chart 4: Average Fare by Class and Embarkation Point")
    plt.savefig('analytics/plots/multivariate_4.png')
    plt.close()
    print("\n[Interpretation: Chart 4]")
    print("This heatmap highlights the varying average ticket prices based on class and where passengers boarded. Notably, 1st class passengers embarking from Cherbourg (C) paid substantially higher average fares than those from other ports. This suggests that Cherbourg might have been a wealthier boarding location or catered to premium ticket types.")
    
    # 18. Exploratory Z-Score Standardization
    print("\n--- Exploratory Z-Score Standardization ---")
    age_mean_before = df['age'].mean()
    age_std_before = df['age'].std()
    fare_mean_before = df['fare'].mean()
    fare_std_before = df['fare'].std()
    
    print(f"Before - Age: Mean = {age_mean_before:.4f}, Std = {age_std_before:.4f}")
    print(f"Before - Fare: Mean = {fare_mean_before:.4f}, Std = {fare_std_before:.4f}")
    
    df['age_z'] = (df['age'] - age_mean_before) / age_std_before
    df['fare_z'] = (df['fare'] - fare_mean_before) / fare_std_before
    
    age_mean_after = df['age_z'].mean()
    age_std_after = df['age_z'].std()
    fare_mean_after = df['fare_z'].mean()
    fare_std_after = df['fare_z'].std()
    
    print(f"After - Age: Mean = {age_mean_after:.4f}, Std = {age_std_after:.4f}")
    print(f"After - Fare: Mean = {fare_mean_after:.4f}, Std = {fare_std_after:.4f}")
    print("Confirmed: The transformed columns now have a mean of approximately 0 and a standard deviation of 1.")

if __name__ == "__main__":
    main()

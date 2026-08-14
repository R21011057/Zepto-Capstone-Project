# Module 2 — Analytics Pipeline (Part A)

This section of the module performs exploratory data analysis (EDA), missing value handling, and multivariate visual storytelling on the Titanic dataset.

## Data Loading & Integrity
- The raw Titanic dataset was fetched exactly once via `sns.load_dataset('titanic')` and saved offline as `titanic.csv`.
- All subsequent steps, including modeling, read strictly from this CSV.

## Missing Value Handling & Justification
- **`deck` (77.22% missing)**: The 'deck' column has over 77% missing values. Imputing such a vast majority of the data would heavily skew any analysis and introduce massive bias. Therefore, we will drop the 'deck' column entirely.
- **`age` (19.87% missing)**: Age has ~20% missing values. We imputed this using the median age to preserve the dataset size without severely impacting the central tendency.
- **`embarked` and `embark_town` (0.22% missing)**: These have <5% missing values. We dropped the specific rows containing these missing values.

## Skewness Interpretation (Fare)
- **Calculated Statistics**: Mean = 32.10, Median = 14.45, Mode = 8.05
- **Interpretation**: The distribution of 'fare' is right-skewed (positive skew), as evidenced by the Mean > Median > Mode ordering. This implies most passengers paid lower fares, while a small number of extreme outliers paid significantly higher fares, pulling the mean to the right.

## Correlation Analysis
- **Top 2 Absolute Off-Diagonal Correlations**:
  1. `pclass` and `fare` (-0.548)
  2. `sibsp` and `parch` (0.415)
- **Interpretation**: The strongest off-diagonal correlation is between pclass and fare (negative). This makes logical sense as a lower class number (1st class) indicates a higher ticket fare. The second strongest is between sibsp and parch (positive), indicating that passengers who traveled with siblings/spouses were also highly likely to travel with parents/children, capturing the presence of families aboard.

## Multivariate Data Story

*(Note: The plots referenced below have been saved as PNG files in the `analytics/plots/` directory during script execution).*

### Chart 1: Survival Rate by Passenger Class and Sex
**Interpretation**: This bar chart illustrates the stark interaction between class, sex, and survival. Females across all classes had a significantly higher survival rate than males, demonstrating the 'women and children first' protocol. Additionally, 1st class females had a near 100% survival rate, while 3rd class males suffered the highest casualties.

### Chart 2: Age Distribution by Survival and Sex
**Interpretation**: This violin plot reveals the age distribution of survivors vs. non-survivors broken down by sex. We can observe a prominent bulge at younger ages (children) for male survivors compared to non-survivors, reiterating the priority given to children. Female distributions are relatively similar whether they survived or not, given their generally high survival rate.

### Chart 3: Age vs. Fare colored by Survival
**Interpretation**: This scatter plot visualizes the relationship between a passenger's age and fare, marked by their survival outcome. The highest fares are heavily clustered with survivors, regardless of age. Conversely, the dense cluster of non-survivors resides in the low-fare, lower-class brackets, further establishing economic status as a primary survival factor.

### Chart 4: Average Fare by Class and Embarkation Point
**Interpretation**: This heatmap highlights the varying average ticket prices based on class and where passengers boarded. Notably, 1st class passengers embarking from Cherbourg (C) paid substantially higher average fares than those from other ports. This suggests that Cherbourg might have been a wealthier boarding location or catered to premium ticket types.

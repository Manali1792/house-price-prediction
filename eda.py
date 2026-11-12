from data_loader import load_data

df = load_data()

# 1. Basic information
print("\n--- Dataset Info ---")
print(df.info())

# 2. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe().T)

# 3. Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Duplicate records
print("\n--- Duplicate Records ---")
print(df.duplicated().sum())

# 5. Target distribution
print("\n--- Target Distribution ---")
print(df["MedHouseVal"].describe())

# 6. Correlation with target
print("\n--- Feature Correlation with Target ---")
print(
    df.corr(numeric_only=True)["MedHouseVal"]
    .sort_values(ascending=False)
)

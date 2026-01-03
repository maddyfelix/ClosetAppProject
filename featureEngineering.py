import pandas as pd

# Load processed data
df = pd.read_parquet("processed_fashion.parquet")

# Define feature columns
feature_columns = ['gender', 'masterCategory', 'subCategory', 'baseColour', 'season', 'usage']

# One-hot encoding
X = pd.get_dummies(df[feature_columns], drop_first=False)

# Reverse one-hot encoding for 'season'
season_cols = [col for col in X.columns if col.startswith('season_')]
X['season'] = X[season_cols].idxmax(axis=1).str.replace('season_', '')
X = X.drop(columns=season_cols)

print(X.head())
print(f"Rows: {X.shape[0]}, Features: {X.shape[1]}")

# Save features for later
X.to_parquet("fashion_features.parquet", index=False)


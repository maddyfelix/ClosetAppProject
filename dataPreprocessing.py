import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO

# Load data
df = pd.read_parquet("train_0.parquet")
print(df.head())
print(f"Number of samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Filter categories
df = df[df["masterCategory"].isin(["Apparel", "Accessories", "Footwear"])]
df = df[df["subCategory"].isin([
    "Topwear", "Bottomwear", "Bags", "Shoes", "Eyewear", "Jewelry",
    "Belts", "Socks", "Dress", "Sandals", "Headwear"
])]

# Keep subCategories with >=120 samples
subcategory_counts = df["subCategory"].value_counts()
valid_subcategories = subcategory_counts[subcategory_counts >= 120].index
df = df[df["subCategory"].isin(valid_subcategories)]

# Replace Boys/Girls → Men/Women
df['gender'] = df['gender'].replace({'Boys': 'Men', 'Girls': 'Women'})

# Quick checks
print(df['gender'].value_counts())
print(df['masterCategory'].value_counts())
print(df['subCategory'].value_counts())
print(df['baseColour'].value_counts())
print(df['season'].value_counts())
print(df['usage'].value_counts())

# Heatmap: MasterCategory by Gender
pivot_table = df.pivot_table(index='gender', columns='masterCategory', aggfunc='size', fill_value=0)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues')
plt.title('Master Category Count by Gender')
plt.tight_layout()
plt.show()

# Gender distribution
gender_counts = df['gender'].value_counts()
plt.style.use('ggplot')
ax = gender_counts.plot(kind='bar', color='skyblue', figsize=(6, 5))
for i, v in enumerate(gender_counts):
    ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
plt.title('Distribution by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=13)
plt.ylabel('Count', fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Image info check
image_bytes = df['image'].iloc[0]['bytes']
img = Image.open(BytesIO(image_bytes))
print("Width:", img.width, "Height:", img.height, "Size:", img.size)
print("Total images:", len(df))

# Save processed dataframe
df.to_parquet("processed_fashion.parquet", index=False)

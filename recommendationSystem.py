import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# Load data
df = pd.read_parquet("processed_fashion.parquet")

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Embedding function
def get_image_embedding(image_dict):
   try:
       image_bytes = image_dict['bytes']
       image = Image.open(BytesIO(image_bytes)).convert("RGB")
       inputs = processor(images=image, return_tensors="pt")
       with torch.no_grad():
           outputs = model.get_image_features(**inputs)
       return outputs[0]
   except Exception:
       return None


tqdm.pandas()
df["embedding"] = df["image"].progress_apply(get_image_embedding)


df = df.reset_index(drop=True)
mask = df["embedding"].notnull()


emb_t = torch.stack([t.detach().cpu().float() for t in df.loc[mask, "embedding"]])
row_idx = df.index[mask].to_list()


torch.save({"emb": emb_t, "row_idx": row_idx}, "clip_embeddings_clean.pt")
df.drop(columns=["image", "embedding"]).to_parquet("fashion_meta.parquet", index=True)
print("✅ Saved clip_embeddings_clean.pt and fashion_meta.parquet")


print("✅ Saved embeddings to clip_embeddings.pt and fashion_with_embeddings.parquet")


# Pick a sample item
sample_row_index = 2
sample_embedding = df["embedding"].iloc[sample_row_index].unsqueeze(0)
sample_item = df.iloc[sample_row_index]


print(f"Recommendations for: {sample_item['productDisplayName']}")


# Show original image
try:
   original_img = Image.open(BytesIO(sample_item['image']['bytes'])).convert("RGB")
   plt.imshow(original_img)
   plt.title(sample_item['productDisplayName'])
   plt.axis('off')
   plt.show()
except Exception as e:
   print("Could not display original:", e)


# Filter similar items
filtered_df = df[
   (df["gender"] == sample_item["gender"]) &
   (df["season"] == sample_item["season"]) &
   (df["usage"] == sample_item["usage"])
]


# Recommend best match per subcategory
print("\nRecommended Products:")
for subcategory in filtered_df["subCategory"].unique():
   if subcategory != sample_item["subCategory"]:
       sub_df = filtered_df[filtered_df["subCategory"] == subcategory]
       if not sub_df.empty:
           embeddings = torch.stack(sub_df["embedding"].tolist())
           similarities = cosine_similarity(sample_embedding, embeddings)
           idx = similarities.argmax()
           rec_item = sub_df.iloc[idx]
           print(f"- {rec_item['productDisplayName']} ({subcategory}) (Similarity: {similarities[0][idx]:.4f})")
           try:
               rec_img = Image.open(BytesIO(rec_item['image']['bytes'])).convert("RGB")
               plt.imshow(rec_img)
               plt.title(rec_item['productDisplayName'])
               plt.axis('off')
               plt.show()
           except:
               pass

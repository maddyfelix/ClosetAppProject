import streamlit as st
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# Initialize session state for page navigation
# ---------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------------------------------------------------------------
# Page and style
# ---------------------------------------------------------------------
st.set_page_config(page_title="Smart Closet", layout="centered")

st.markdown(
   """
   <style>
       /* App background */
       [data-testid="stAppViewContainer"] {
           background-color: #f2ebee;
           font-family: 'Arial', sans-serif;
       }

       /* Header */
       [data-testid="stHeader"] {
           background: none;
       }

       /* Sidebar */
       [data-testid="stSidebar"] {
           background-color: #F7E4EB;
           font-family: 'Arial', sans-serif;
       }

       /* ==============================
          SELECTBOX (DROPDOWN) STYLING
          ============================== */
       div[data-baseweb="select"] > div {
           background-color: #F7E4EB;
           border-radius: 8px;
           border: 1px solid #e5a9c5;
       }

       div[data-baseweb="select"] span {
           color: #5a2a3c;
           font-weight: 500;
       }

       ul[data-baseweb="menu"] {
           background-color: #F7E4EB;
           border-radius: 8px;
       }

       ul[data-baseweb="menu"] li {
           background-color: #F7E4EB;
           color: #5a2a3c;
       }

       ul[data-baseweb="menu"] li:hover {
           background-color: #e5a9c5;
       }

       div[data-baseweb="select"] > div:focus {
           box-shadow: none;
       }

       /* ==============================
          HOMEPAGE BUTTON STYLING
          ============================== */
       div.stButton > button {
           background-color: #F7E4EB;
           color: #5a2a3c;
           border: 1px solid #e5a9c5;
           border-radius: 8px;
           padding: 0.5em 1em;
           font-weight: 500;
       }

       div.stButton > button:hover {
           background-color: #e5a9c5;
           color: white;
       }
   </style>
   """,
   unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# Helper: strict image loader
# ---------------------------------------------------------------------
def open_image_safe(image_obj, name="(unknown)"):
   if image_obj is None:
       err = f"No image data for {name}"
       st.error(err)
       raise RuntimeError(err)
   if not isinstance(image_obj, dict):
       err = f"Image for {name} has wrong type: {type(image_obj)}"
       st.error(err)
       raise RuntimeError(err)
   if "bytes" not in image_obj:
       err = f"Missing 'bytes' key in image for {name}"
       st.error(err)
       raise RuntimeError(err)
   try:
       img = Image.open(BytesIO(image_obj["bytes"]))
       img.verify()
       img = Image.open(BytesIO(image_obj["bytes"]))
       return img
   except Exception as e:
       err = f"Could not open image for {name}: {e}"
       st.error(err)
       raise RuntimeError(err)

# ---------------------------------------------------------------------
# Load data and model
# ---------------------------------------------------------------------
@st.cache_resource
def load_clip():
   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
   model.eval()
   return model, processor

@st.cache_data
def load_data():
   df = pd.read_parquet("fashion_meta.parquet")
   pack = torch.load("clip_embeddings_clean.pt", weights_only=True)
   emb_list = [None] * len(df)
   for i, t in zip(pack["row_idx"], pack["emb"]):
       emb_list[i] = t
   df["embedding"] = emb_list

   original_df = pd.read_parquet("processed_fashion.parquet")[["productDisplayName", "image"]]
   original_df = original_df.drop_duplicates(subset="productDisplayName", keep="first")
   df = df.merge(original_df, on="productDisplayName", how="left", validate="m:1")
   return df

# ---------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------
def recommend_outfit(df, idx, model, processor):
   sample_item = df.iloc[idx]
   sample_embedding = sample_item["embedding"].unsqueeze(0)

   filtered_df = df[
       (df["gender"] == sample_item["gender"]) &
       (df["season"] == sample_item["season"]) &
       (df["usage"] == sample_item["usage"])
   ]

   recs = []
   for subcategory in filtered_df["subCategory"].unique():
       if subcategory != sample_item["subCategory"]:
           sub_df = filtered_df[filtered_df["subCategory"] == subcategory]
           if not sub_df.empty:
               embeddings = torch.stack(sub_df["embedding"].tolist())
               similarities = cosine_similarity(sample_embedding, embeddings)
               idx_best = similarities.argmax()
               rec_item = sub_df.iloc[idx_best]
               recs.append((rec_item, float(similarities[0][idx_best])))
   return sample_item, recs

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Default closet", "Custom upload"],
    index=["Home", "Default closet", "Custom upload"].index(st.session_state.page),
    key="sidebar_page"
)
st.session_state.page = page  # keep session state in sync

# ---------------------------------------------------------------------
# Homepage
# ---------------------------------------------------------------------
if st.session_state.page == "Home":
   st.title("Welcome to CoCo Closet")
   st.markdown("""
   <p style="font-size:18px;">
   Then you can browse the default closet for ready-to-use recommendations<br>
   or upload your own clothing items to get personalized sugestions...<br>
   </p>
   """, unsafe_allow_html=True)

   col1, col2 = st.columns(2)
   with col1:
       if st.button("Browse Default Closet"):
           st.session_state.page = "Default closet"
   with col2:
       if st.button("Upload Your Own Item"):
           st.session_state.page = "Custom upload"

# ---------------------------------------------------------------------
# Default closet tab
# ---------------------------------------------------------------------
elif st.session_state.page == "Default closet":

   st.subheader("Select an Item")

   df = load_data()
   model, processor = load_clip()

   # Step 1: choose gender
   genders = sorted(df["gender"].dropna().unique())
   selected_gender = st.selectbox("Gender:", genders)

   if selected_gender:
       # Step 2: choose category
       categories = sorted(df.loc[df["gender"] == selected_gender, "subCategory"].dropna().unique())
       selected_category = st.selectbox("Clothing category:", categories)

       if selected_category:
           # Step 3: choose item
           filtered_items = df.loc[
               (df["gender"] == selected_gender) &
               (df["subCategory"] == selected_category),
               "productDisplayName"
           ].tolist()
           selected_item = st.selectbox("Item:", filtered_items)

           if selected_item:
               idx = df.index[df["productDisplayName"] == selected_item][0]

               with st.spinner("Generating recommendations..."):
                   sample_item, recs = recommend_outfit(df, idx, model, processor)

               # Display selected item
               st.subheader("Selected Item")
               col1, col2 = st.columns([1, 2])
               with col1:
                   img = open_image_safe(sample_item["image"], sample_item["productDisplayName"])
                   st.image(img.convert("RGB"), caption=sample_item["productDisplayName"], use_container_width=True)
               with col2:
                   st.write(f"**Category:** {sample_item['masterCategory']} → {sample_item['subCategory']}")
                   st.write(f"**Gender:** {sample_item['gender']}")
                   st.write(f"**Season:** {sample_item['season']}")
                   st.write(f"**Usage:** {sample_item['usage']}")

               # Recommendations
               st.subheader("Recommended Outfit Matches")
               if not recs:
                   st.info("No recommendations found for this item.")
               else:
                   for rec_item, sim in recs:
                       sim_text = ""
                       if sim > 0.9:
                           sim_text = "<span style='background-color:#2ecc71; color:white; padding:3px 6px; border-radius:4px;'>Very high!</span>"
                       elif sim <= 0.9 and sim >= 0.7:
                           sim_text = "<span style='background-color:#a3cb38; color:black; padding:3px 6px; border-radius:4px;'>High</span>"
                       elif sim < 0.7 and sim >= 0.5:
                           sim_text = "<span style='background-color:#f1c40f; color:black; padding:3px 6px; border-radius:4px;'>Moderate</span>"
                       else:
                           sim_text = "<span style='background-color:#e74c3c; color:white; padding:3px 6px; border-radius:4px;'>Low</span>"

                       cols = st.columns([1, 2])
                       with cols[0]:
                           rec_img = open_image_safe(rec_item["image"], rec_item["productDisplayName"])
                           st.image(rec_img.convert("RGB"), caption=rec_item["productDisplayName"], use_container_width=True)
                       with cols[1]:
                           st.write(f"**{rec_item['productDisplayName']}**")
                           st.write(f"Category: {rec_item['subCategory']}")
                           st.write(f"Recommendation strength: {sim_text}", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Custom upload tab
# ---------------------------------------------------------------------
else:
   st.subheader("Upload a custom item")

   uploaded_file = st.file_uploader(
       "Upload a clothing image",
       type=["png", "jpg", "jpeg"]
   )

   if uploaded_file is not None:
       custom_img = Image.open(uploaded_file)
       st.image(
           custom_img,
           caption="Uploaded item",
           use_container_width=True
       )
       st.info("Custom logic will go here later (embedding + recommendations).")

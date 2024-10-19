import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models import extract_image_features, get_text_embedding, apply_synonym_mapping

# Preprocessing images for feature extraction
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Match description to images from the dataset
def match_description_to_images(description):
    from config import person_csv_path, samples_folder
    person_df = pd.read_csv(person_csv_path, delimiter=';')

    # Apply feature extraction
    text_features = extract_text_features(description)
    image_features = extract_features_for_samples(samples_folder)
    
    # Match text features to image features
    text_embedding = get_text_embedding(text_features)
    image_embeddings = np.array(list(image_features.values()))
    top_indices, similarities = compare_embeddings(text_embedding, image_embeddings)
    
    # Return top 3 matches
    return get_top_matches(person_df, top_indices, similarities, image_features)

# Helper to filter CSV
def filter_person_df_with_images(person_df, samples_folder):
    existing_images = set(os.path.splitext(f)[0] for f in os.listdir(samples_folder) if f.endswith('.jpg'))
    return person_df[person_df['id'].astype(str).isin(existing_images)]

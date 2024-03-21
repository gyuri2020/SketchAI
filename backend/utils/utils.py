from clip_emb.utils.feature_extractor import FeatureExtractor
import pandas as pd
import os
from typing import List
import numpy as np
from PIL import Image
import json

backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_path = os.path.dirname(backend_path)
feature_extractor = FeatureExtractor()
montage_vector_df = pd.read_parquet(f"{repo_path}\\data\\vectors\\montage_vectors.parquet")
image_path_format = repo_path + "\\data\\Images\\montage\\{num}.png"
sketch_path_format = repo_path + "\\data\\Images\\org_sketch\\{num}.png"
json_path_format = repo_path + "\\data\\json\\H\\{num}.json"
sketch_info_dict = {}


def calculate_similarity(query_vector: List[float], montage_vectors: List[List[float]]):
    vector1 = np.array(query_vector).astype(float).reshape(1, -1)
    vector2 = np.array(montage_vectors).astype(float)

    similarity = vector1 @ vector2.T
    return similarity

def get_sketch_for_image(im):
    im_pil = im['composite']

    vector = feature_extractor.get_image_feature(image_pil=im_pil)
    vector_list = montage_vector_df["image_embedding"].to_list()
    similarity_scores = calculate_similarity(vector, vector_list)
    top_idx = np.argmax(similarity_scores)
    print(top_idx)
    top_montage_id = montage_vector_df.iloc[top_idx]["montage_id"]
    print(top_montage_id)
    image_path = image_path_format.format(num=top_montage_id)
    sketch_path = sketch_path_format.format(num=top_montage_id)
    json_path = json_path_format.format(num=top_montage_id)

    img = Image.open(image_path)
    sketch = Image.open(sketch_path)
    global sketch_info_dict
    with open(json_path, 'r', encoding='cp949') as f:
        sketch_info_dict = json.load(f)
        sketch_info_dict = sketch_info_dict.get("description", {})
        if "org_id" in sketch_info_dict.keys():
            del sketch_info_dict["org_id"]

    return img, sketch

def get_sketch_for_text(text):
    

    vector = feature_extractor.get_text_features([text])[0].tolist()
    vector_list = montage_vector_df["image_embedding"].to_list()
    similarity_scores = calculate_similarity(vector, vector_list)
    top_idx = np.argmax(similarity_scores)
    print(top_idx)
    top_montage_id = montage_vector_df.iloc[top_idx]["montage_id"]
    print(top_montage_id)
    image_path = image_path_format.format(num=top_montage_id)
    sketch_path = sketch_path_format.format(num=top_montage_id)
    json_path = json_path_format.format(num=top_montage_id)

    img = Image.open(image_path)
    sketch = Image.open(sketch_path)
    global sketch_info_dict
    with open(json_path, 'r', encoding='cp949') as f:
        sketch_info_dict = json.load(f)
        sketch_info_dict = sketch_info_dict.get("description", {})
        if "org_id" in sketch_info_dict.keys():
            del sketch_info_dict["org_id"]

    return img, sketch


def show_sketch_info(category):
    global sketch_info_dict
    print(sketch_info_dict)
    print(category)
    print(sketch_info_dict.get(category, {}))
    return sketch_info_dict.get(category, {})
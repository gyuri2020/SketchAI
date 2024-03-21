from clip_emb.utils.feature_extractor import FeatureExtractor
import pandas as pd
from tqdm import tqdm


montage_path_format = "C:\\Users\\82109\\Desktop\\sketchAI\\SketchAI\\data\\Images\\montage\\{num}.png"
montage_id_range = range(20, 13343)
feature_extractor = FeatureExtractor()
vector_file_path = "C:\\Users\\82109\\Desktop\\sketchAI\\SketchAI\\data\\vectors\\montage_vectors.parquet"

vector_list = []
for montage_id in tqdm(montage_id_range):

    if montage_id % 100 == 0:
        # Save every 100 montages
        df = pd.DataFrame(vector_list)
        df.to_parquet(vector_file_path)
        print(f"Saved {montage_id} montages")

    try:
        image_path = montage_path_format.format(num=montage_id)
        image_embedding = feature_extractor.get_image_feature(image_path)
        vector_list.append({"montage_id": montage_id, "image_embedding": image_embedding})
    except:
        continue

df = pd.DataFrame(vector_list)
df.to_parquet(vector_file_path)
print("Done")


    

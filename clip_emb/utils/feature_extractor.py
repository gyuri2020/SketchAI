from multilingual_clip import pt_multilingual_clip
import transformers
from typing import List
import torch
import open_clip
import requests
from PIL import Image


class FeatureExtractor:

    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
    nlp_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    vision_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")

    def get_text_features(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.nlp_model.forward(texts, self.tokenizer)
        return embeddings
    

    def get_image_feature(self, image_path: str=None, image_pil: Image=None) -> List[float]:
        if image_path:
            img = Image.open(image_path)
        else:
            img = image_pil
        img = self.preprocess(img).unsqueeze(0)
        embedding = self.vision_model.encode_image(img).reshape(-1).tolist()
        return embedding
    

if __name__ == "__main__":
    fe = FeatureExtractor()
    print(fe.get_text_features(["hello", "world"]))
    print(fe.get_image_feature("https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"))
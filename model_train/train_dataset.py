import os
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import glob
import re
import json
import yaml



ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageCaptionDataset(Dataset):

    def __init__(self):
        super().__init__()
        json_dict_path = "data/json/H/"
        montage_dict_path = "data/Images/montage"
        self.json_files = self.get_all_file_paths(json_dict_path)
        self.montage_files = self.get_all_file_paths(montage_dict_path)
        json_nums = [self.parse_num_from_path(path) for path in self.json_files]
        montage_nums = [self.parse_num_from_path(path) for path in self.montage_files]
        self.data_nums = list(set(json_nums) & set(montage_nums))
        self.json_files = [f"{json_dict_path}/{num}.json" for num in self.data_nums]
        self.montage_files = [
            f"{montage_dict_path}/{num}.png" for num in self.data_nums
        ]
        print("example")
        print(f"json files: {self.json_files[2]}")
        print(f"montage files: {self.montage_files[2]}")

    def __len__(self):
        return len(self.data_nums)

    def __getitem__(self, idx):
        image = self._get_image(idx)
        captions = self._get_caption(idx)
        return {"image_id": f"{idx}", "image": image, "caption": captions}

    def _get_image(self, idx):
        image_path = self.montage_files[idx]
        image = Image.open(image_path)
        image = image.convert("RGBA")
        return image

    def _get_caption(self, idx):
        file = self.json_files[idx]
        data = self.read_korean_json(file)

        description = data["description"]
        captions = []
        for key, value in description.items():
            if type(value) == dict and value.get("description"):
                captions.append(value["description"])

        return captions

    def get_all_file_paths(self, directory):
        return glob.glob(directory + "/**", recursive=True)

    def parse_num_from_path(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else None

    def read_korean_json(self, file_path):
        with open(file_path, "r", encoding="cp949") as f:
            data = json.load(f)
        return data


class ImageCaptionCollator(object):
    def __init__(self, processor, image_size=224, max_caption_length=77):
        # 77 is the max length of captions in the dataset
        self.processor = processor
        self.image_size = image_size
        self.max_caption_length = max_caption_length

    def __call__(self, batch):
        image_ids = []
        images = []
        captions = []
        for row in batch:
            caps = row["caption"]
            for i, cap in enumerate(caps):
                image_ids.append(f"{row['image_id']}_{i}")
                images.append(row["image"])
                captions.append(cap)

        # image preprocessing: feature extractor defaults
        # caption preprocessing: pad/truncate + tensor
        inputs = self.processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_caption_length,
            truncation=True,
        )

        return inputs, image_ids

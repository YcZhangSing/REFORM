import json
import math
import os
import re
from pathlib import Path
from random import random as rand

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, resize


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


LABEL_TO_OPTION = {
    "orig": "A. No.",
    "face_swap": "B. Image: Face swap; Text: No.",
    "face_attribute": "C. Image: Face attribute; Text: No.",
    "full_gene": "D. Image: Whole generated; Text: No.",
    "bg_rep": "E. Image: Inpainted background; Text: No.",
    "face_swap&text_swap": "F. Image: Face swap; Text: Fully rewritten.",
    "face_attribute&text_swap": "G. Image: Face attribute; Text: Fully rewritten.",
    "full_gene&text_swap": "H. Image: Whole generated; Text: Fully rewritten.",
    "bg_rep&text_swap": "I. Image: Inpainted background; Text: Fully rewritten.",
    "text_swap": "J. Image: No; Text: Fully rewritten.",
}

OPTIONS = list(LABEL_TO_OPTION.values())

DESCRIBE_TEMPLATE = (
    "<image>The following are multiple choice questions about fake news detection.\n"
    "The text caption of news is: "
)
DESCRIBE_QUESTION_10OPTS = (
    ".\n The image and text should not be manipulated. Question: Is there any "
    "manipulation in the image or text of this news?\n"
    "A. No.\n"
    "B. Image: Face swap; Text: No.\n"
    "C. Image: Face attribute; Text: No.\n"
    "D. Image: Whole generated; Text: No.\n"
    "E. Image: Inpainted background; Text: No.\n"
    "F. Image: Face swap; Text: Fully rewritten.\n"
    "G. Image: Face attribute; Text: Fully rewritten.\n"
    "H. Image: Whole generated; Text: Fully rewritten.\n"
    "I. Image: Inpainted background; Text: Fully rewritten.\n"
    "J. Image: No; Text: Fully rewritten.\n"
)
FACE_LOCATE_PROMPT = (
    "If the face is manipulated, locate the manipulated face in the image and "
    "append the results to your selected option.\nThe answer is:"
)


def pre_caption(caption, max_words):
    if not caption:
        return "No text"

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    caption = emoji_pattern.sub("", str(caption))
    caption = re.sub(r"[#;:~()*<>/\\^_{}\[\]|=+]", "", caption)
    caption = caption.replace("<person>", "person")
    caption = re.sub(r"\s{2,}", " ", caption).strip()

    words = caption.split(" ")
    if len(words) > max_words:
        caption = " ".join(words[:max_words])
    return caption


def infer_dataset_root(json_path):
    path = Path(json_path).resolve()
    parent = path.parent
    if parent.parent.name in {"train", "val", "test"}:
        return parent.parent.parent
    return parent


def resolve_image_path(image_path, dataset_root=None):
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    root = Path(dataset_root) if dataset_root else Path.cwd()
    return str((root / path).resolve())


def load_rom_json(json_path, dataset_root=None):
    json_path = Path(json_path)
    root = Path(dataset_root) if dataset_root else infer_dataset_root(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "image" in item:
            item["image"] = resolve_image_path(item["image"], root)
    return data


class ROMDatasetForTraining(Dataset):
    def __init__(self, split, data, max_words=30, image_res=224):
        self.name = "ROM"
        self.data = data
        self.max_words = max_words
        self.image_res = image_res
        self.is_train = split == "train"

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_bbox(bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)

    @staticmethod
    def denormalize_fake_image_box_xyxy(fake_image_box, image_width, image_height):
        center_x, center_y, w, h = fake_image_box
        abs_center_x = center_x * image_width
        abs_center_y = center_y * image_height
        abs_w = w * image_width
        abs_h = h * image_height
        x1 = abs_center_x - abs_w / 2
        y1 = abs_center_y - abs_h / 2
        x2 = abs_center_x + abs_w / 2
        y2 = abs_center_y + abs_h / 2
        return round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)

    def __getitem__(self, index):
        ann = self.data[index]
        label = ann["fake_cls"].replace("text_attribute", "text_swap")
        image_path = ann["image"]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        has_bbox = False
        fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        mask = np.zeros((self.image_res, self.image_res, 1))

        if any(keyword in label for keyword in ["face_swap", "face_attribute"]):
            try:
                x, y, w, h = self.get_bbox(ann["fake_image_box"])
                has_bbox = True
            except Exception:
                has_bbox = False

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                image = hflip(image)
                do_hflip = True
            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)

        if has_bbox:
            if do_hflip:
                x = (width - x) - w

            x = self.image_res / width * x
            w = self.image_res / width * w
            y = self.image_res / height * y
            h = self.image_res / height * h

            mask_x = math.floor(x)
            mask_y = math.floor(y)
            mask_w = math.ceil(w)
            mask_h = math.ceil(h)
            mask[mask_y : mask_y + mask_h, mask_x : mask_x + mask_w, :] = 1

            center_x = x + 0.5 * w
            center_y = y + 0.5 * h
            fake_image_box = torch.tensor(
                [center_x / self.image_res, center_y / self.image_res, w / self.image_res, h / self.image_res],
                dtype=torch.float,
            )

        caption = pre_caption(ann["text"], self.max_words)
        reason = pre_caption(ann.get("Internvl_out_think"), 205)
        question = "<ROM>" + DESCRIBE_TEMPLATE + caption + DESCRIBE_QUESTION_10OPTS + FACE_LOCATE_PROMPT
        answer = LABEL_TO_OPTION[label]

        if has_bbox:
            x1, y1, x2, y2 = self.denormalize_fake_image_box_xyxy(fake_image_box, width, height)
            answer += f"Fake area<loc_{int(x1)}><loc_{int(y1)}><loc_{int(x2)}><loc_{int(y2)}>"

        return image, question, answer, fake_image_box, reason, image_path


class ROMDatasetForEvaluation(ROMDatasetForTraining):
    def __getitem__(self, index):
        image, question, answer, fake_image_box, reason, image_path = super().__getitem__(index)
        caption = pre_caption(self.data[index]["text"], self.max_words)
        return image, question, answer, fake_image_box, image_path, caption


# Backward-compatible names used by the original training scripts.
fullMDSMv2Dataset_10opts_2decoder = ROMDatasetForTraining
fullMDSMv2Dataset_10opts_2decoder_forEVA = ROMDatasetForEvaluation


"""
Image + caption dataset for the tiny text-conditioned avatar diffusion model.

This replaces the old BasicDataset (image + segmentation mask) from your
U-Net segmentation project. The key differences:

  - The "target" is no longer a pixel mask, it's a tokenized caption.
  - No multiprocessing scan for unique mask values is needed.
  - Preprocessing normalizes images to [-1, 1] (standard for DDPM noise
    prediction), not [0, 1] as in the segmentation preprocess().
  - A `split_ids` list restricts the dataset to a given split (train /
    val / test / ood_compositional), so the compositional split defined
    in your split file is respected everywhere.
"""
from pathlib import Path
import sys
sys.path.append("..")

import logging
import numpy as np
import torch
from PIL import Image
from functools import partial
from os.path import splitext
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable, Tuple
import csv
from tokenizer import Tokenizer

def load_data(filename) -> Image.Image:
    return Image.open(filename).convert("RGB")

def parse_attribute_legend(legend_path: str) -> Dict[Tuple[str, str], str]:
    """
    Parses the attribute LEGEND csv — e.g. "cartoon_image_attribute_labels.csv" —
    which maps a numeric attribute code to its human-readable text label.
    Expected columns (header row required): attribute, value, text_label

        attribute,value,text_label
        eye_color,0,blue
        eye_color,1,green
        hair_style,0,short
        hair_style,1,long
        gender,0,boy
        gender,1,girl

    Returns {(attribute, value): text_label}, both keyed as strings so they
    match whatever numeric-code representation the per-image file uses.
    No attribute name is hardcoded — whatever "attribute" values appear in
    the csv are the components the caption builder will later use.
    """
    legend: Dict[Tuple[str, str], str] = {}
    with open(legend_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attribute = row["attribute"].strip()
            value = str(row["value"]).strip()
            text_label = row["text_label"].strip()
            legend[(attribute, value)] = text_label
    return legend


def parse_image_attributes(
        image_attribute_path: str, legend: Dict[Tuple[str, str], str]
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Parses cartoon_image_attributes.csv — wide format, one row per image,
    header names = attribute names, first column = path relative to
    images_dir (subfolder included):

        filename,eye_angle,hair_color,...
        0/cs11556364481883459966.jpg,2,2,...

    Every numeric code is resolved to its text label via `legend`. No
    attribute is hardcoded — whatever columns exist after "filename"
    become the components used by build_deterministic_caption.

    Returns:
        metadata:   {image_id: {attribute: text_label}}
        id_to_path: {image_id: relative_path}   -- e.g. "0/cs...9966.jpg",
                    used to open the file directly, no filesystem scan needed.
    """
    metadata: Dict[str, Dict[str, str]] = {}
    id_to_path: Dict[str, str] = {}

    with open(image_attribute_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        path_col = fieldnames[0]  # "filename"
        attribute_cols = fieldnames[1:]  # every other column is an attribute

        for row in reader:
            rel_path = row[path_col].strip()
            image_id = splitext(Path(rel_path).name)[0]  # "cs11556364481883459966"
            id_to_path[image_id] = rel_path

            attrs_for_image: Dict[str, str] = {}
            for attribute in attribute_cols:
                value = str(row[attribute]).strip()
                text_label = legend.get((attribute, value))
                if text_label is not None:
                    attrs_for_image[attribute] = text_label
            metadata[image_id] = attrs_for_image

    return metadata, id_to_path


def build_deterministic_caption(
        attributes: Dict[str, str],
        subject_component: str = "gender",
        component_order: Optional[List[str]] = None,
) -> str:
    """
    Turn a per-image {component: value} dict (as produced by
    parse_image_attributes) into a deterministic caption, by looping
    over whatever components are present — no attribute is special-cased.

    subject_component: the one component used as the sentence subject
                        (e.g. "a <boy> with ..."); everything else becomes
                        a "<value> <component>" clause, in a for loop.
    component_order:   optional explicit ordering for the clauses, to keep
                        captions deterministic/reproducible across runs.
                        Defaults to alphabetical order of the components
                        actually present for this image.

    Example:
        {"gender": "boy", "eye_color": "blue", "hair_style": "short",
         "proportions": "exaggerated", "accessory": "glasses"}
        -> "a boy with blue eye color, short hair style, exaggerated
            proportions and glasses accessory"
    """
    attrs = dict(attributes)
    subject = attrs.pop(subject_component, "avatar")

    order = component_order or sorted(attrs.keys())
    clauses = []
    for component in order:
        if component not in attrs:
            continue
        value = attrs[component]
        if not value:
            continue
        clauses.append(f"{value} {component.replace('_', ' ')}")

    caption = f"a {subject}"
    if clauses:
        if len(clauses) == 1:
            caption += f" with {clauses[0]}"
        else:
            caption += " with " + ", ".join(clauses[:-1]) + f" and {clauses[-1]}"
    return caption


class AvatarDataset(Dataset):
    def __init__(
            self,
            images_dir: str,
            attribute_legend_path: str,
            image_attribute_path: str,
            tokenizer: Tokenizer,
            split_ids: Optional[List[str]] = None,
            image_size: int = 32,
            subject_component: str = "gender",
            component_order: Optional[List[str]] = None,
            attribute_fn: Optional[Callable[[Dict[str, str]], str]] = None,
    ):
        """
            images_dir:             folder with avatar images, filenames "<id>.png" etc.
            attribute_legend_path:  csv "cartoon_image_attribute_labels.csv" mapping
                                    (attribute, numeric value) -> text_label. Columns:
                                     attribute, value, text_label.
            image_attribute_path:   csv "cartoon_image_attribute.csv" mapping each image
                                    to its numeric attribute codes (long or wide format,
                                    auto-detected — see parse_image_attributes).
            tokenizer:              a CaptionTokenizer whose vocab was already built on
                                    the TRAIN split captions only.
            split_ids:              list of ids belonging to this split (train/val/
                                    test/ood). If None, uses every id found on disk
                                    (only sensible for quick smoke tests).
            subject_component/component_order: forwarded to build_deterministic_caption
                                    (see there) to keep caption phrasing consistent.
            attribute_fn:           override the caption-building function entirely
                                    if you want a different template; defaults to
                                    build_deterministic_caption.
        """
        self.images_dir = images_dir
        self.image_size = image_size
        self.tokenizer = tokenizer

        legend = parse_attribute_legend(attribute_legend_path)
        self.metadata, self.id_to_path = parse_image_attributes(image_attribute_path, legend)

        available = set(self.metadata.keys())
        self.ids = sorted(available if split_ids is None else (available & set(split_ids)))
        if not self.ids:
            raise RuntimeError(
                f"No matching (image, attribute-label) pairs found for the given split "
                f"(csv had {len(available)} entries)."
            )
        logging.info(f"Dataset: {len(self.ids)} examples")

        self.attribute_fn = attribute_fn or partial(
            build_deterministic_caption,
            subject_component=subject_component,
            component_order=component_order,
        )

        self.captions: Dict[str, str] = {
            i: self.attribute_fn(self.metadata[i]) for i in self.ids
        }

        self.captions = {i: self.attribute_fn(self.metadata[i]) for i in self.ids}
        logging.info(f"Example caption: {next(iter(self.captions.values()))}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img: Image.Image, size: int):
        pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
        img = np.asarray(pil_img).astype(np.float32)
        img = img.transpose((2, 0, 1))  # C,H,W
        img = img / 127.5 - 1.0  # -> [-1, 1], standard for DDPM eps-prediction
        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = self.images_dir +"/"+ self.id_to_path[name]   # e.g. images_dir / "0/cs...9966.jpg"
        img = load_data(img_path)
        img = self.preprocess(img, self.image_size)

        caption = self.captions[name]
        token_ids = self.tokenizer.encode(caption)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "input_ids": torch.as_tensor(token_ids, dtype=torch.long),
            "caption": caption,  # kept for logging/inspection, not used by the model
            "id": idx,
        }
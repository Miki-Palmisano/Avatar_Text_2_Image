"""
Pre-processa TUTTE le immagini (resize + normalizzazione) una volta sola,
e le salva come array .npy — così AvatarDataset.__getitem__ carica un
array già pronto invece di aprire/decodificare/resizare il JPEG ad ogni
accesso, per ogni epoca.

Da lanciare UNA VOLTA prima del training (richiede qualche minuto su
100k immagini, ma lo paghi una sola volta — non ad ogni epoca).

Uso:
    python3 utils/cache_loading.py \
        --images_dir dataset/cartoonset100k \
        --attribute_legend_path dataset/cartoon_image_attributes_labels.csv \
        --image_attribute_path dataset/cartoon_image_attributes.csv \
        --cache_dir cache_64 \
        --image_size 64
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_loading import load_data, parse_attribute_legend, parse_image_attributes, AvatarDataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--attribute_legend_path", required=True)
    p.add_argument("--image_attribute_path", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                    help="float16 dimezza lo spazio su disco; float32 se preferisci precisione piena")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)

    images_dir = Path(args.images_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    legend = parse_attribute_legend(args.attribute_legend_path)
    metadata, id_to_path = parse_image_attributes(args.image_attribute_path, legend)

    dtype = np.float16 if args.dtype == "float16" else np.float32

    n_done, n_skipped, n_failed = 0, 0, 0
    for image_id, rel_path in tqdm(id_to_path.items(), desc="Caching immagini"):
        out_path = cache_dir / f"{image_id}.npy"
        if out_path.exists():
            n_skipped += 1   # già processata in un run precedente, non rifare lavoro
            continue

        try:
            img = load_data(images_dir / rel_path)
            arr = AvatarDataset.preprocess(img, size=args.image_size)  # stesso preprocess usato dal dataset
            np.save(out_path, arr.astype(dtype))
            n_done += 1
        except Exception as e:
            logging.warning(f"Fallita {image_id} ({rel_path}): {e}")
            n_failed += 1

    logging.info(f"Fatto. Nuove: {n_done}, già presenti: {n_skipped}, fallite: {n_failed}")
    logging.info(f"Cache salvata in: {cache_dir}")


if __name__ == "__main__":
    main()
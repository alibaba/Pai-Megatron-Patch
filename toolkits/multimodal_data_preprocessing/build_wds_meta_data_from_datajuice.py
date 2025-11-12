import json
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm
import yaml
from transformers.audio_utils import load_audio_as
import cv2
import pickle

from webdataset.writer import default_handlers, add_handlers, imageencoder
import webdataset as wds
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory


def generate_configs(path: EPath, split, shuffle_tars=True, num_workers=32):
    path = path.absolute()
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
    split_parts_ratio = [("train", split[0]), ("val", split[1]), ("test", split[2])]
    split_parts_patterns = None
    
    # NOTE: generate .info.yaml and split.yaml
    _ = BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        tar_index_only=False,
        shuffle_seed=42 if shuffle_tars else None,
        workers=num_workers,
    )

    # NOTE: dump dataset.yaml
    metadata = {
        '__class__': 'ChatMLWebdataset',
        '__module__': 'megatron_patch.data.multimodal_dataset',
        'field_map': {
            'audios': 'mp3s',
            'imgs': 'jpgs',
            'videos': 'mp4s',
            'conversation': 'json'
        }
    }
    with open(os.path.join(path.url, '.nv-meta', 'dataset.yaml'), 'w') as f:
        yaml.safe_dump(metadata, f)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--dataset-root', required=True, type=str)
    argparser.add_argument('--json', default='dataset.json', type=str)
    argparser.add_argument('--max-samples-per-tar', default=1, type=float)
    argparser.add_argument('--train-split', default=9, type=float)
    argparser.add_argument('--val-split', default=1, type=float)
    argparser.add_argument('--test-split', default=0, type=float)
    args = argparser.parse_args()

    split=[args.train_split, args.val_split, args.test_split]
    generate_configs(EPath(args.dataset_root), split)
    print(f"Configurations Generated")
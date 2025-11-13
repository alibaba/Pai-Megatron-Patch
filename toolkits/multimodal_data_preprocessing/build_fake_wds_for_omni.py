# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from argparse import ArgumentParser
import yaml
import cv2
import pickle

from webdataset.writer import default_handlers, add_handlers, imageencoder
import webdataset as wds
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory

def audioencoder(value):
    return value

def convert(dataset_dir, max_count=10000):

    current_dir = os.getcwd()

    output = os.path.join(current_dir, dataset_dir)

    if not os.path.exists(output):
        os.mkdir(output)

    add_handlers(default_handlers, "jpgs", lambda data: pickle.dumps([imageencoder(d, "jpg") for d in data]))
    add_handlers(default_handlers, "mp4s", lambda data: pickle.dumps([[imageencoder(d, "jpg") for d in video] for video in data]))
    add_handlers(default_handlers, "mp3s", lambda data: pickle.dumps([audioencoder(d) for d in data]))

    with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=max_count) as shard_writer:
        for entry in range(2):
            image_datas = []
            image_datas.append(cv2.imread(os.path.join(current_dir, "australia.jpg"), cv2.IMREAD_UNCHANGED))

            video_datas = []
            video_datas.append([cv2.imread(os.path.join(current_dir, "australia.jpg"), cv2.IMREAD_UNCHANGED), cv2.imread(os.path.join(current_dir, "australia.jpg"), cv2.IMREAD_UNCHANGED)])

            second_per_grid_ts = [0.5, 0.5]

            audio_datas = []
            with open(os.path.join(current_dir, "glass-breaking-151256.mp3"), "rb") as aud_file:
                aud_data = aud_file.read()
                audio_datas.append(aud_data)

            conversations = [
                {"role": "system", "content": "你是个有用无害的助手"},
                {"role": "user", "content": "<image>图片中是什么，<video>视频中是什么"},
                {"role": "assistant", "content": "图片中是一个大象，视频中是一只小狗在草地上奔跑"},
                {"role": "user", "content": "<audio>语音说了什么"},
                {"role": "assistant", "content": "今天天气真好呀"}
            ]

            sample = {
                "__key__": str(entry),
                "mp3s": audio_datas,
                "mp4s": video_datas,
                "jpgs": image_datas,
                "json": json.dumps({'conversations': conversations,
                    'second_per_grid_ts': second_per_grid_ts}
                ).encode("utf-8"),
            }
            shard_writer.write(sample)

    print(f"Dataset successfully converted to wds")
    return output

def generate_configs(path: EPath, split, shuffle_tars=True, num_workers=32):
    path = path.absolute()
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
    split_parts_ratio = [("train", split[0]), ("val", split[1]), ("test", split[2])]
    split_parts_patterns = None

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
    argparser.add_argument('--output-dir', required=True, type=str)
    argparser.add_argument('--max-samples-per-tar', default=1, type=float)
    argparser.add_argument('--train-split', default=9, type=float)
    argparser.add_argument('--val-split', default=1, type=float)
    argparser.add_argument('--test-split', default=0, type=float)
    args = argparser.parse_args()

    output_dir = convert(args.output_dir, max_count=args.max_samples_per_tar)
    split=[args.train_split, args.val_split, args.test_split]
    generate_configs(EPath(output_dir), split)
    print(f"Web Datasets Generated")
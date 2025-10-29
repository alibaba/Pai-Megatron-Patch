import json
import os
import sys
import webdataset as wds
from tqdm import tqdm
from transformers.audio_utils import load_audio_as

def convert(output_dir):

    """
    conversations = [
                {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
            {"role": "user", "content": [
                    {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
                ]},
        ]
    """

    current_dir = os.getcwd()

    output = os.path.join(current_dir, output_dir)

    if not os.path.exists(output):
        os.mkdir(output)

    with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=10000) as shard_writer:
        for entry in range(1000):
            with open(os.path.join(current_dir, "australia.jpg"), "rb") as img_file:
                image_data = img_file.read()

            aud = load_audio_as("glass-breaking-151256.mp3", return_format='base64')
            sample = {
                "__key__": str(entry),
                "jpg": image_data,
                "aud": aud,
                "json": 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'.encode("utf-8"),
            }
            shard_writer.write(sample)

    print(f"Dataset successfully converted to wds")


if __name__ == '__main__':
    convert(sys.argv[1])
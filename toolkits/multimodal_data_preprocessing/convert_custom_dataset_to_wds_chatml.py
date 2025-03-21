import json
import os
import webdataset as wds
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import cv2
from webdataset.writer import default_handlers, add_handlers, imageencoder
import pickle

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory

def convert(dataset_dir, json_name, sort_function=sorted, max_count=10000):
    """
        Here we provide an example to convert llava-pretrain dataset to ChatMLSample
    """
    # Paths to the dataset files
    json_file = os.path.join(dataset_dir, json_name)
    output = os.path.join(dataset_dir, 'wds')

    if not os.path.exists(output):
        os.mkdir(output)

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # custom webdataset ShardWriter Encoder
    add_handlers(default_handlers, "jpgs", lambda data: pickle.dumps([imageencoder(d, "jpg") for d in data]))
    add_handlers(default_handlers, "videos", lambda data: pickle.dumps([[imageencoder(d, "jpg") for d in video] for video in data]))

    has_idx = None
    with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=max_count) as shard_writer:
        for idx, entry in enumerate(tqdm(data)):
            # NOTE: read a dataset in sharegpt format
            image_datas = []
            for image in entry.pop('images', []):
                image_datas.append(cv2.imread(os.path.join(dataset_dir, image), cv2.IMREAD_UNCHANGED))
            
            video_datas = []
            second_per_grid_ts = []
            for video in entry.pop('videos', []):
                video_noext, _ = os.path.splitext(video)
                frame_folder = os.path.join(dataset_dir, video_noext)
                # NOTE: we implicitly require a `${frame_folder}.json`` file containing fps rates of each video
                # otherwise fps will be regarded as `1` by default.
                if os.path.exists(frame_folder + '.json'):
                    with open(frame_folder + '.json', 'r') as f:
                        fps = float(json.load(f)['fps'])
                else:
                    fps = 2.0

                frames = []
                for frame in sort_function(os.listdir(frame_folder)):
                    frames.append(cv2.imread(os.path.join(frame_folder, frame), cv2.IMREAD_UNCHANGED))
                
                if len(frames) % 2 == 1:
                    frames = frames[:-1]
                video_datas.append(frames)
                second_per_grid_ts.append(1 / fps)


            if has_idx is None:
                has_idx = 'id' in entry
            assert has_idx == ('id' in entry), "All entries should either all contain idx or not."
            
            sample = {
                "__key__": entry.pop('id', str(idx)), 
                "jpgs": image_datas,
                'videos': video_datas,
                "json": json.dumps({
                    'conversations': entry['conversations'],
                    'second_per_grid_ts': second_per_grid_ts
                }).encode("utf-8"),
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
        '__module__': 'megatron_patch.data.energon.chatml',
        'field_map': {
            'imgs': 'jpgs',
            'videos': 'videos',
            'conversation': 'json'
        }
    }
    with open(os.path.join(path.url, '.nv-meta', 'dataset.yaml'), 'w') as f:
        yaml.safe_dump(metadata, f)
    
if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--dataset-root', required=True, type=str)
    argparser.add_argument('--json', default='dataset.json', type=str)
    argparser.add_argument('--max-samples-per-tar', default=10000, type=float)
    argparser.add_argument('--train-split', default=9, type=float)
    argparser.add_argument('--val-split', default=1, type=float)
    argparser.add_argument('--test-split', default=0, type=float)
    args = argparser.parse_args()


    output_dir = convert(args.dataset_root, args.json, max_count=args.max_samples_per_tar)
    print(f"Generating Configurations")
    # NOTE: split_ratio: train/val/test
    split=[args.train_split, args.val_split, args.test_split]
    generate_configs(EPath(output_dir), split)
    print(f"Configurations Generated")
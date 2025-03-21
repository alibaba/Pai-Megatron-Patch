import os
import glob
import tarfile
import json
import cv2
from multiprocessing import Pool

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

def find_json_files(dataset_root):
    root_path = Path(dataset_root).resolve()
    json_files = list(root_path.rglob("*.json"))
    jsonl_files = list(root_path.rglob("*.jsonl"))
    
    all_files = json_files + jsonl_files
    relative_paths = [p.relative_to(root_path) for p in all_files]
    return [str(p) for p in relative_paths]

def extract_video_frames(
    dataset_root: str, 
    video_paths: list, 
    time_interval: float = 1.0,
):
    for rel_path in video_paths:
        input_path = os.path.join(dataset_root, rel_path)
        output_subdir, _ = os.path.splitext(input_path)
        os.makedirs(output_subdir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Video not opened: {input_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        interval_frames = max(1, int(fps * time_interval)) 
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % interval_frames == 0:
                filename = f"frame_{current_frame:06}.jpg"
                save_path = os.path.join(output_subdir, filename)
                cv2.imwrite(save_path, frame)

            current_frame += 1

        with open(output_subdir + '.json', 'w') as f:
            json.dump({
                'fps': str(fps / interval_frames)
            }, f)


def process(dataset_root, output_file, interval=1.0, num_workers: int=32, video_token='<image>'):
    json_or_jsonl = (
        glob.glob(os.path.join(dataset_root, '*.json')) + 
        glob.glob(os.path.join(dataset_root, '*.jsonl'))
    )
    
    full_data = []
    
    args_list = []
    for file in find_json_files(dataset_root):
        rel_to_dir, _ = os.path.split(file)
        file = os.path.join(dataset_root, file)
        try:
            with open(file, 'r') as f:
                data = json.load(f)
        except:
            with open(file, 'r') as f:
                data = [json.loads(f) for l in f.readlines()]

        print(f'processing {file}')
        for d in tqdm(data):
            if isinstance(d, list):
                assert len(d) == 1
                d = d[0]
            if "image" in d:
                d['images'] = [os.path.join(rel_to_dir, d.pop('image'))]
            if "video" in d:
                d['videos'] = [os.path.join(rel_to_dir, d.pop('video'))]
                for v in d['videos']:
                    args_list.append((dataset_root, [v], interval))
            
            for c in d['conversations']:
                c['value'] = c['value'].replace(video_token, '<video>')
            full_data.append(d)
    
    pool = Pool(32)
    it = pool.istarmap(extract_video_frames, args_list)
    for _ in tqdm(it, total=len(args_list)):
        pass

    with open(os.path.join(dataset_root, output_file), 'w') as f:
        json.dump(full_data, f)

def extract_video(dataset_root):
    # extract all .tar.gz to the split folder
    splits = os.listdir(dataset_root)
    for split in splits:
        p = os.path.join(dataset_root, split)
        if not os.path.isdir(p):
            continue
        files = [f for f in os.listdir(p) if f.endswith('.tar.gz')]
        for f in files:
            with tarfile.open(os.path.join(p, f), 'r:gz') as tar:
                tar.extractall(path=p)


if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument("--dataset-root", type=str, required=True, help="The root of LLaVA-Video-178K dataset")
    argparser.add_argument("--time-interval", type=float, default=1.0, help="The time interval to extract frame from videos")
    argparser.add_argument("--output-json", type=str, default='dataset.json', help="Filename of the merged json dataset")
    argparser.add_argument("--skip-extraction", action='store_true')
    argparser.add_argument("--video-token", type=str, default='<image>', help="The default video token in LLaVA-Video-178K is <image> instead of <video>")

    args = argparser.parse_args()
    
    if not args.skip_extraction:
        print("video extraction starting")
        extract_video(args.dataset_root)
        print("video extraction finished")
    process(args.dataset_root, args.output_json, interval=args.time_interval, video_token=args.video_token)
    
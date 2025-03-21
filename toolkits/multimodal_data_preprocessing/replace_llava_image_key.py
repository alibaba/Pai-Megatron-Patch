import os
import json

from argparse import ArgumentParser


def process(
    in_file,
    out_file,
):
    d = os.path.dirname(out_file)
    os.makedirs(d, exist_ok=True)

    try:
        with open(in_file, 'r') as f:
            data = json.load(f)
    except:
        with open(in_file, 'r') as f:
            data = [json.loads(f) for l in f.readlines()]
    for i, sample in enumerate(data):
        if isinstance(sample, list):
            assert len(sample) == 1
            data[i] = sample[0]
            if "image" in data[i]:
                data[i]['images'] = [data[i].pop('image')]

    with open(out_file, 'w') as f:
        json.dump(data, f)    

if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument("--input-file", type=str, required=True)
    argparser.add_argument("--output-file", type=str, default='dataset.json')

    args = argparser.parse_args()
    process(args.input_file, args.output_file)
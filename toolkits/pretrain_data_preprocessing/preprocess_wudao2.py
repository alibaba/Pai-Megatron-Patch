import argparse
import codecs
import json
import multiprocessing
import os.path
import re
from glob import glob
from tqdm import tqdm


def clean_text(raw):
    httpcom = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@'
        r'.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
    raw = httpcom.sub('', raw)

    space = re.compile(r' +')
    raw = space.sub(' ', raw)

    fil = re.compile(
        u'[^0-9a-zA-Z\u4e00-\u9fa5.， ,\\-。%'
        u'《*》/•、&＆(—)（+）：？!！“”·]+', re.UNICODE)
    raw = fil.sub('', raw)
    return raw.strip()


def run_preprocess(input_fp, output_fp, target_key):
    with codecs.open(output_fp, 'w', encoding='utf8') as json_file:
        with open(input_fp) as f:
            try:
                file_json = json.load(f)
            except ValueError:
                file_json = {}
            for obj in file_json:
                text = obj['content']
                text = clean_text(text)
                di = {
                    'instruction': "",
                    'input': "",
		            target_key : text
                }
                dumped_di = json.dumps(di, ensure_ascii=False)
                json_file.write(dumped_di + '\n')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input-dir',
                        '-input_dir',
                        '-i',
                        help='folder name of checkpoint files',
                        required=True)

    parser.add_argument('--output-dir',
                        '-output_dir',
                        '-o',
                        help='folder name of checkpoint files',
                        required=True)

    parser.add_argument('--target-key',
                        '-target_key',
                        '-k',
                        type=str,
                        default='content',
                        help='target_key',
                        )

    parser.add_argument('--num-processes',
                        '-num_processes',
                        '-p',
                        type=int,
                        default=None,
                        help='Number of processes')

    args = parser.parse_args()
    po = multiprocessing.Pool(args.num_processes)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for input_file in tqdm(glob(args.input_dir + '/*.json')):
        fn = input_file.split('/')[-1]
        output_file = os.path.join(args.output_dir, fn)
        po.apply_async(func=run_preprocess, args=(input_file, output_file, args.target_key))
    po.close()
    po.join()
    print('done')


if __name__ == '__main__':
    main()

import sys
import pandas
import json

if len(sys.argv) < 2:
    print("请提供文件路径作为参数")
    sys.exit(1)

samples = []
file_path = sys.argv[1]
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        fin = json.load(f)
except Exception:
    fin = []
    with open(file_path, 'r', encoding='utf-8') as f:
        fin = [json.loads(d) for d in f.readlines()]
assert isinstance(fin, list)
for jdict in fin:
    instruct = jdict["instruction"]
    input = jdict["input"]
    output = jdict["output"]
    samples.append(instruct+input+output)

pd = pandas.Series(samples).map(len)
print(pd.describe())



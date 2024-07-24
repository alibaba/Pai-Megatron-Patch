import sys
import pandas
import json

if len(sys.argv) < 2:
    print("请提供文件路径作为参数")
    sys.exit(1)

samples = []
file_path = sys.argv[1]
with open(file_path) as f:
    for line in f:
        jdict = json.loads(line)
        instruct = jdict["instruction"]
        input = jdict["input"]
        output = jdict["output"]
        samples.append(instruct+input+output)

pd = pandas.Series(samples).map(len)
print(pd.describe())



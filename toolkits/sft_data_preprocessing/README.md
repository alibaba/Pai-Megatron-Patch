## 微调数据预处理

用于微调的数据组织方式为每一行对应一条样本，用json字典表示，如下所示：
```shell
{"instruction": "读下面的段落，找出一个比喻句子。", "input": "\"我的烦恼长出了翅膀，飞走进了天空\"", "output": "比喻：我的烦恼长出了翅膀，飞走进了天空。"}
```
同时可以下载我们提供的样例微调数据集，如下所示：
```bash
mkdir /mnt/workspace/qwen-datasets
cd /mnt/workspace/qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/qwen_sft.json
```

#### 代码准备
前往[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)开源网站获取Megatron模型训练工具Pai-Megatron-Patch源代码并拷贝到工作目录/mnt/workspace/下。
```bash
# 开源网站获取训练代码
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

#### 统计数据集长度样本分布

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
python sample_stats.py /mnt/workspace/qwen-datasets/qwen_sft.json
```
输出结果如下所示：
```bash
count    50151.000000            # 样本总数
mean       113.771131            # 样本平均长度
std         90.073800            # 标准差
min          6.000000            # 样本最小长度
25%         51.000000            # 25%的样本长度小于51
50%         93.000000            # 50%的样本长度小于93
75%        150.000000            # 75%的样本长度小于150
max       2458.000000            # 样本最大长度
```

#### 制作MMAP格式预训练数据集
mmap数据是一种预先执行tokenize处理的数据格式，可以极大减少训练微调过程中等待数据读入的时间，当数据量极大时，优势显著。

在DSW的Terminal中进入代码目录：/mnt/workspace/Pai-Megatron-Patch/toolkits/sft_data_preprocessing。查看run_build_idxmap_data_for_sft.sh脚本内容。里面有5个启动参数需要在运行时输入，具体参数列表如下：
```
input_data_path=$1                # 设置输入文件路径
tokenizer=$2                      # 设置分词器
seq_len=$3                        # 设置训练用的序列长度
output_data_path=$4               # 设置输出文件路径  
load_dir=$5                       # 设置HF模型的路径

```
运行示例如下所示：
```bash
sh run_build_idxmap_data_for_sft.sh \
/mnt/workspace/qwen-datasets/qwen_sft.json \
Qwen2Tokenizer \
256 \
/mnt/workspace/qwen-datasets/mmap_qwen2_sft_datasets \
/mnt/workspace/qwen-ckpts/Qwen2-0.5B
```
脚本执行完成后，qwen-datasets文件夹里有2个名字相同后缀不同的mmap文件，您需要在训练脚本中使用/mnt/workspace/qwen-datasets/mmap_qwen2_sft_datasets_text_document来访问它们：
```bash
qwen-datasets
   ├── mmap_qwen2_sft_datasets_text_document.bin
   └── mmap_qwen2_sft_datasets_text_document.idx
```

#### Sequence Packing

目前Pai-Megatron-Patch中的部分模型(LLaMA3.1, Qwen-2等)已支持基于mmap格式的Sequence-Packing训练，为此，您首先需要按照下列步骤准备打包后的数据集

1. 下载json数据集
2. 打包SFT样本

为了实现Sequence Packing，在json格式的基础上，您需要决定每个Sequence包含的样本。具体而言，您需要进一步将训练时被packing的多个json文本放至同一列以实现打包。
例如，下列代码将qwen_sft数据集按顺序分组，使得每组packing后的长度不超过2048。为了提高Packing的效率，您也可以自行拓展脚本来使用其他策略

```python

from transformers import AutoTokenizer
import json
import pdb
tokenizer = AutoTokenizer.from_pretrained("/mnt/qwen-ckpts/Qwen2-0.5B")
seqlen = 2048
example_len_map = dict()
with open("/mnt/qwen-datasets/qwen_sft.json") as f:
    for line in f:
        line = line.strip()
        example = json.loads(line)
        input = example["instruction"]+example['input']
        output = example['output']
        input_ids = tokenizer(input, add_special_tokens=False)['input_ids']
        output_ids = tokenizer(output, add_special_tokens=False)['input_ids']
        example_len_map[line] = len(input_ids) + len(output_ids) + 3

example_group = []
len_group = []
with open("/mnt/qwen-datasets/packed_qwen_sft.json", 'w', encoding='utf8') as f:
    for example, len in example_len_map.items():
        example = json.loads(example)
        if sum(len_group) <= seqlen:
            example_group.append(example)
            len_group.append(len)
        else:
            last_example = example_group[-1]
            last_example_len = len_group[-1]
            assert last_example_len <= seqlen
            json_string = json.dumps(example_group[:-1], ensure_ascii=False)
            f.write(json_string+"\n")
            example_group = [last_example]+[example]
            len_group = [last_example_len]+[len]
```

3. 在完成json格式数据打包后，运行下列命令来获得用于LLaMA3.1 SFT的packed mmap数据集
```
bash run_build_packed_idxmap_sft_dataset.sh \
/workspace/llama-datasets/packed_qwen_sft.json \
LLama3Tokenizer \
2048 \
/workspace/llama-datasets/packed_sft_dataset \
/workspace/Meta-Llama-3.1-8B
```

4. 微调训练时，设置`SFT=true`，同时设置环境变量`MP_SFT_PACKING=true`即可使用Sequence Packing。

#### 小规模预处理数据下载试用
为方便用户试用，我们也提供了已经处理好的小规模数据，可直接下载使用
```bash
cd /mnt/workspace/qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen2_sft_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen2_sft_datasets_text_document.idx
```
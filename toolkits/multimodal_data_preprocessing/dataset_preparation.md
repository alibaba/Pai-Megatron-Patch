# 准备基于ShareGPT格式的Energon多模态数据集

当前Qwen2-VL/Qwen2.5-VL支持特定格式的复杂多模态样本的训练，您可按照下述流程将您的数据集转换为我们支持的格式。

## 原始数据

在转换前，你可能需要自行将数据集转换为**sharegpt格式**，sharegpt格式的示例如下:
```json
[
  {
    "conversations": [
        {
            "from": "human",
            "value": "<image>human instruction<image>"
        },
        {
            "from": "gpt",
            "value": "model response"
        },
        {
            "from": "human",
            "value": "<video><video>human instruction"
        },
        {
            "from": "gpt",
            "value": "model response"
        }
    ],
    "images": [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
    ],
    "videos": [
        "path/to/video1.mp4",
        "path/to/video2.mp4"
    ]
  },
  {
    // another sample ...
  }
]
```
其中，`images`与`videos`列表保存所有图像/视频的原始路径，且依次与对话中的`<image>`与`<video>`标记对应。

## 抽帧
在训练前，您需要使用DataJuicer等工具将数据集中的视频转换为一系列帧图像。

以`path/to/video1.mp4`为例，假设其保存在`dataset_root/path/to/video1.mp4`, 最终您导出的帧应当保存在 `dataset_root/path/to/video1/` 这一文件夹。此外，您需要保证帧图像的时间顺序与文件名字典序顺序一致。
推荐文件名示例如下
```
00001.jpg # frame 1
00002.jpg # frame 2
...
```

通过引入动态分辨率采样以及绝对时间对齐技术，Qwen2.5-VL能更好地支持对于不同FPS的视频的理解。为了启用这一特性，对于每个视频文件，在抽帧的同时，您同时需要保存所抽帧的帧率到json文件中。例如，对于保存到`dataset_root/path/to/video1/`的帧，您需要将帧率按下列格式保存到`dataset_root/path/to/video1.json`中。
```
{
    "fps": "2.0 (该视频导出帧的帧率)"
}
```

对于LLaVA-Video-178K等llava格式视频数据集，我们提供了简易脚本将其处理成sharegpt格式供小规模测试使用。对于大规模任务，我们仍推荐使用专门的数据处理工具对其进行抽帧。
运行以下命令，下载并处理LLaVA-Video-178K的部分数据(NextQA)。
```
cd /mnt/llava-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/LLaVA-Video-178K-demo.tar
tar -xvf LLaVA-Video-178K-demo.tar

cd /workspace/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python build_llava_frame_dataset.py \
    --dataset-root /mnt/llava-datasets/LLaVA-Video-178K \
    --time-interval 0.5 # 每0.5秒保存一帧，导出帧帧率为2.0 (实际可能有舍入，以保存的json文件为准)

```

然后您可以对`/mnt/llava-datasets/LLaVA-Video-178K`调用`convert_custom_dataset_to_wds_chatml.py`制作训练数据集。

### 其他

对于llava格式的图像数据集，您可以直接使用下述脚本处理jsonl，即可使用`convert_custom_dataset_to_wds_chatml.py`制作训练数据集。

```
# replace `image` key with `images`
python replace_llava_image_key.py \
    --input-file your_raw_dataset.json_or_jsonl \
    --output-file dataset.json

```

## 转换
假设数据集目录文件结构如下:
```
dataset_root/
-   dataset.json
-   ...
```

运行以下命令将上述准备好的json数据集转换为训练格式, 并存储到`dataset_root/wds`文件夹
```
python toolkits/pretrain_data_preprocessing/convert_custom_dataset_to_wds_chatml.py \
--dataset-root dataset_root \
--json dataset.json \
--train-split 9 \
--val-split 1 \
--test-split 0
```
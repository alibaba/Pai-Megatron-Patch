# 准备Qwen2-VL多模态数据集

当前Qwen2-VL支持特定格式的复杂多模态样本的训练，您可按照下述流程将您的数据集转换为Qwen2-VL的支持格式。

## 原始数据

在转换前，你可能需要自行将数据集转换为**sharegpt格式**，示例如下:
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

## 转换
假设数据集目录文件结构如下:
```
dataset_root/
-   dataset.json
-   ...
```

运行以下命令将上述准备好的数据集转换为Qwen2-VL训练使用的webdataset, 并存储到`dataset_root/wds`文件夹
```
python toolkits/pretrain_data_preprocessing/convert_custom_dataset_to_wds_chatml.py \
--dataset-root dataset_root \
--json dataset.json \
--train-split 9 \
--val-split 1 \
--test-split 0
```
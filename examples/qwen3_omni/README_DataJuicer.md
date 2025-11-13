## DataJuicer介绍

DataJuicer支持多模态数据的转换、清洗和处理，并支持大规模数据的分布式加速。

更多功能请参考开源项目：https://github.com/datajuicer/data-juicer

目前支持的数据清洗/处理功能请参考：https://github.com/datajuicer/data-juicer/blob/main/docs/Operators.md

PAI产品侧使用文档请参考：[快速提交DataJuicer任务参考文档](https://help.aliyun.com/zh/pai/user-guide/quickly-submit-a-datajuicer-task?spm=a2c4g.11174283.help-menu-search-30347.d_0)


## 数据

**原始数据格式如下**：

```json
{"messages": [{"role": "user", "content": "浙江的省会在哪？"}, {"role": "assistant", "content": "浙江的省会在杭州。"}, {"role": "user", "content": "<image>图中是杭州什么地方"}, {"role": "assistant", "content": "西溪湿地。"}]}
{"messages": [{"role": "user", "content": "<image><image>两张图片有什么区别"}, {"role": "assistant", "content": "前一张是小猫，后一张是小狗"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
{"messages": [{"role": "user", "content": "<audio>语音说了什么"}, {"role": "assistant", "content": "今天天气真好呀"}], "audios": ["/xxx/x.mp3"]}
{"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "<image>图片中是什么，<video>视频中是什么"}, {"role": "assistant", "content": "图片中是一个大象，视频中是一只小狗在草地上奔跑"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```

**目标格式如下：**

webdataset格式，字段mapping如下：

```yaml
field_mapping:
    # __key__: 'id'  # id没有会生成一个随机id
    # text: 'text'    # str
    jpgs: 'images'  # List[bytes]
    mp3s: 'audios'  # List[bytes]
    mp4s: 'frames'  # List[List[bytes]]
    json: 'messages'  # messages
```

其中 **mp4s: 'frames'** 表示：原始数据中videos字段在DataJuicer里做抽帧，帧数据保存在frames字段下。保存成webdataset格式时 frames字段 会保存在"mp4s"关键字下。


## DataJuicer使用

### 数据准备
DataJuicer输入数据集文件参考：

</path/to/your/data.jsonl>

```json
{"messages": [{"role": "user", "content": "<image>图片中是什么，<video>视频中是什么, <audio>音频中是什么"}, {"role": "assistant", "content": "同时也能提高肺部的通气功能和氧气的利用效率，长期坚持有氧运动可以降低心血管疾病的风险，改善血脂水平，控制体重，增强免疫系统功能。"}], "audios": ["/path/to/audios/027_19_M_JKYS_021.wav"], "videos": ["/path/to/videos/027_19_M_JKYS_021.mp4"], "images": ["/path/to/images/027_19_M_JKYS_021/keyframe_0.jpg", "/path/to/images/027_19_M_JKYS_021/keyframe_1.jpg"]}
{"messages": [{"role": "user", "content": "<image>图片中是什么，<video>视频中是什么, <audio>音频中是什么"}, {"role": "assistant", "content": "如肌肉线条的塑造，体能的增强等，都能让个人对自己的身体形象和能力，有更积极的认识。"}], "audios": ["/path/to/audios/027_19_M_JKYS_119.wav"], "videos": ["/path/to/videos/027_19_M_JKYS_119.mp4"], "images": ["/path/to/images/027_19_M_JKYS_119/keyframe_0.jpg"]}
{"messages": [{"role": "system", "content": "You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice."}, {"role": "user", "content": "<image>图片中是什么，<video>视频中是什么, <audio>音频中是什么"}, {"role": "assistant", "content": "降低血压和心率水平，第一，运动可以增强心肺功能，提高心脏的泵血效率，从而降低血压和心率水平，通过有氧运动如慢跑、游泳、骑自行车等。"}], "audios": ["/path/to/audios/027_19_M_JKYS_050.wav"], "videos": ["/path/to/videos/027_19_M_JKYS_050.mp4"], "images": ["/path/to/images/027_19_M_JKYS_050/keyframe_0.jpg", "/path/to/images/027_19_M_JKYS_050/keyframe_1.jpg", "/path/to/images/027_19_M_JKYS_050/keyframe_2.jpg"]}
......
```

### 自定义算子准备
1. data-juicer支持text文本字段（string）处理，但不支持messages list dict格式的文本处理，因为暂时不对messages字段做处理 所以这里直接dump成文本。更标准的多模态格式的转换可以参考：[https://github.com/modelscope/data-juicer/blob/main/tools/fmt_conversion/multimodal/README_ZH.md](https://github.com/modelscope/data-juicer/blob/main/tools/fmt_conversion/multimodal/README_ZH.md)
2. 恢复messages json格式导出，并过滤关键帧小于等于1的样本（Pai-Megatron-Patch训练要求视频帧数大于1）。


自定义脚本：

预处理 preprocess_data.py  转换文本字段（也可以前期数据准备时做好处理）

```python
import json

from data_juicer.ops.base_op import OPERATORS, Mapper

OP_NAME = "preprocess_mapper"


@OPERATORS.register_module(OP_NAME)
class PreprocessMapper(Mapper):
    """"""

    _batched_op = True

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def process_single(self, sample):
        # data-juicer支持text文本字段，但不支持messages list dict格式
        # 因为暂时不对messages字段做处理 所以这里直接dump成文本. 
        # 更标准的多模态格式的转换可以参考：https://github.com/modelscope/data-juicer/blob/main/tools/fmt_conversion/multimodal/README_ZH.md
        sample[self.text_key] = json.dumps(sample[self.text_key])

        return sample

```

后处理 postprocess_data.py 适配Pai-Megatron-Patch的训练格式

```python
import json

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields

_FRAMES_NUM_KEY = "frames_num"

OP_NAME = "postprocess_filter"

@OPERATORS.register_module(OP_NAME)
class PostprocessFilter(Filter):
    """"""

    _batched_op = True

    def __init__(self, frame_key, min_frame_num=1, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.frame_key = frame_key
        self.min_frame_num = min_frame_num
    
    def compute_stats_single(self, sample, context=False):
        """
        Compute stats for the sample which is used as a metric to decide
        whether to filter this sample.

        :param sample: input sample.
        :param context: whether to store context information of intermediate
            vars in the sample temporarily.
        :return: sample with computed stats
        """
        # restore to json format for messages field
        sample[self.text_key] = json.loads(sample[self.text_key])

        # check if it's computed already
        if _FRAMES_NUM_KEY in sample[Fields.stats]:
            return sample

        # there is no frames in this sample
        if self.frame_key not in sample or not sample[self.frame_key]:
            sample[Fields.stats][_FRAMES_NUM_KEY] = []
            return sample

        sample[Fields.stats][_FRAMES_NUM_KEY] = [
            len(frames) for frames in sample[self.frame_key]
        ]

        return sample

    def process_single(self, sample):
        """
        For sample level, sample --> Boolean.

        :param sample: sample to decide whether to filter
        :return: true for keeping and false for filtering
        """
        frames_nums = sample[Fields.stats][_FRAMES_NUM_KEY]
        return bool(frames_nums) and all(x > self.min_frame_num for x in frames_nums)

```

### yaml文件准备
demo.yaml

```yaml
# global parameters
project_name: 'to-webdataset'  # 项目名称
dataset_path: '/path/to/your/data.jsonl'  # 数据集路径，支持文件夹

export_path: './outputs/data/wds/'  # 数据导出保存的路径

text_keys: 'messages'  # 文本字段
video_key: 'videos'
image_key: 'images'
audio_key: 'audios'
image_special_token: '<image>'  # 图像分隔符
audio_special_token: '<audio>'
video_special_token: '<video>'
eoc_special_token: '<|__dj__eoc|>'

ray_address: auto    
executor_type: ray  # 启动ray分布式

skip_op_error: false  # 容错配置，调试阶段可以置为false，运行阶段可以置为true容错

custom_operator_paths:  # 注册自定义算子
  - /path/to/postprocess_data.py
  - /path/to/preprocess_data.py

export_type: 'webdataset'  # 导出数据格式
# extra args for reconstructing webdataset format
export_extra_args:
  field_mapping:
    # __key__: 'id'  # id没有会生成一个随机id
    # text: 'text'    # str
    json: 'messages'  # json meta
    jpgs: 'images'  # List[bytes]
    mp3s: 'audios'  # List[bytes]
    mp4s: 'frames'  # List[List[bytes]]

# process schedule
# a list of several process operators with their arguments
process:
  - preprocess_mapper:
  - video_extract_frames_mapper:
      frame_sampling_method: 'all_keyframes'  # 抽帧方式
      output_format: 'bytes'
      # frame_dir: './outputs/data/frames/'
      frame_key: 'frames'  # 帧保存字段
  - postprocess_mapper:
      frame_key: 'frames'  # 帧保存字段
      min_frame_num: 1

```



### 运行

DataJuicer运行的所有数据和算子配置都保存在yaml文件中。

```shell
dj-process --config demo.yaml
```

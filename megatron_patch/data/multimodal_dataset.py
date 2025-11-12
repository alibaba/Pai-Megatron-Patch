# NOTE: add a license
import warnings
import pickle
import torch
import re
import io
import numpy as np
import torchaudio

from dataclasses import dataclass
from typing import List, Union

from webdataset.autodecode import Decoder, imagehandler
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory

@dataclass
class ChatMLSample(Sample):
    """multi-turn complex samples with images, audios and videos"""
    imgs: List[torch.Tensor]
    audios: List[torch.Tensor]
    videos: List[List[torch.Tensor]]
    conversation: str # JSON string of GPT-format conversations

class videohandler:
    def __init__(self, imagespec):
        """Create an video handler.

        :param imagespec: short string indicating the type of decoding
        """
        self.extensions = ['jpgs', 'mp4s']
        self.extensions_mapping = {
            "jpgs": "jpg",
            "mp4s": "jpg"
        }
        self.image_handler = imagehandler(imagespec)

    def __call__(self, key, data):
        """Perform nested image decoding.

        :param key: file name extension
        :param data: binary data
        """    
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        data = pickle.loads(data)
        key = self.extensions_mapping[extension]
        if extension.lower() == 'jpgs':
            data = [self.image_handler(key, d) for d in data]
        else:
            data = [[self.image_handler(key, d) for d in video] for video in data]
        return data

class audiohandler:

    def __init__(self):
        """Create an audio handler.
        """
        self.extensions = ['wavs', 'mp3s']

    def __call__(self, key, data):
        """Perform audio decoding.

        :param key: file name extension
        :param data: binary data
        """
        extension = re.sub(r".*[.]", "", key)
        if extension not in self.extensions:
            return None

        data_list = pickle.loads(data)
        audio_list = []
        for data in data_list:
            audio_list.append(torchaudio.load(io.BytesIO(data)))
        return audio_list

     
class ChatMLWebdataset(DefaultDecoderWebdatasetFactory[ChatMLSample]):
    __sample_type__ = ChatMLSample

    def __init__(self, path: EPath, *, auto_decode:bool =True, **kwargs):
        super().__init__(path, auto_decode=auto_decode, **kwargs)  
        if auto_decode:
            self._decoder = Decoder(
                [
                    imagehandler(self.image_decode),
                    audiohandler(),
                    videohandler(self.image_decode),
                ]
            )
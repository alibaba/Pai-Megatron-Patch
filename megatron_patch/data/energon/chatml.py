# NOTE: add a license
import warnings
import pickle
import torch
import re

from dataclasses import dataclass
from typing import List, Union

from webdataset.autodecode import Decoder, imagehandler
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory

@dataclass
class ChatMLSample(Sample):
    """multi-turn complex samples with images and videos"""
    imgs: List[torch.Tensor]
    videos: List[List[torch.Tensor]]
    conversation: str # JSON string of GPT-format conversations

class NestedImagesHandler:
    def __init__(self, imagespec):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        """
        self.extensions = ['jpgs', 'videos']
        self.extensions_mapping = {
            "jpgs": "jpg",
            "videos": "jpg"
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
        
class ChatMLWebdataset(DefaultDecoderWebdatasetFactory[ChatMLSample]):
    __sample_type__ = ChatMLSample

    def __init__(self, path: EPath, *, auto_decode:bool =True, **kwargs):
        super().__init__(path, auto_decode=auto_decode, **kwargs)
        if auto_decode:
            self._decoder = Decoder(
                [
                    imagehandler(self.image_decode),
                    NestedImagesHandler(self.image_decode),
                    self._video_decoder,
                ]
            )

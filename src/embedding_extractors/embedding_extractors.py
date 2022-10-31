from abc import ABC, abstractmethod

import torch.nn as nn
from torchvision import transforms


class BaseEmbeddingExtractor(ABC, nn.Module):
    def __int__(self, model, input_size, mean, std):
        self.model = model
        self.input_size = input_size
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = transforms.functional.resize(x, size=self.input_size)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = transforms.functional.normalize(x,
                                            mean=self.mean,
                                            std=self.std)
        x = self.call_model(x)
        return x

    @abstractmethod
    def call_model(self, x):
        pass


class ClipEmbeddingExtractor(BaseEmbeddingExtractor):
    def call_model(self, x):
        x = self.encoder(x.half())
        return x


class OpenClipEmbeddingExtractor(BaseEmbeddingExtractor):
    def call_model(self, x):
        x = self.encoder.encode_image(x)
        return x

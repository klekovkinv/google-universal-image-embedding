import argparse
import os

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from embedding_extractors.embedding_extractors import (
    ClipEmbeddingExtractor, OpenClipEmbeddingExtractor
)


FOLDERS = ('train', 'val')
CLIP_INPUT_SIZE = [336, 336]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
OPEN_CLIP_INPUT_SIZE = [224, 224]
OPEN_CLIP_MEAN = [0.5, 0.5, 0.5]
OPEN_CLIP_STD = [0.5, 0.5, 0.5]


def load_clip_model(path):
    with open(path, 'rb') as opened_file:
        model = torch.jit.load(path, map_location="cuda:0").visual
    return model


def load_open_clip_model(name, suffix):
    model, _, _ = open_clip.create_model_and_transforms(
        name, pretrained=suffix)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-path', type=str,
                        default='data/models/ViT-L-14-336px.pt',
                        help='Path to Clip weights')
    parser.add_argument('--open-clip-name', type=str,
                        default='ViT-L-14',
                        help='OpenClip model name')
    parser.add_argument('--open-clip-suffix', type=str,
                        default='laion2b_s32b_b82k',
                        help='OpenClip model suffix')
    parser.add_argument('--input', type=str,
                        default='data/ggl_dataset',
                        help='Path to input images folder')
    parser.add_argument('--clip-output', type=str,
                        default='data/clip_embeddings',
                        help='Path to Clip output folder')
    parser.add_argument('--open-clip-output', type=str,
                        default='data/open_clip_embeddings',
                        help='Path to OpenClip output folder')
    args = parser.parse_args()

    # models
    clip_model = load_clip_model(path=args.clip_path)
    open_clip_model = load_open_clip_model(name=args.open_clip_name,
                                           suffix=args.open_clip_suffix)

    # embedding extractors
    clip_embedding_extractor = ClipEmbeddingExtractor(
        model=clip_model,
        input_size=CLIP_INPUT_SIZE,
        mean=CLIP_MEAN,
        std=CLIP_STD).cuda().eval()
    open_clip_embedding_extractor = OpenClipEmbeddingExtractor(
        model=open_clip_model,
        input_size=OPEN_CLIP_INPUT_SIZE,
        mean=OPEN_CLIP_MEAN,
        std=OPEN_CLIP_STD).cuda().eval()

    # transforms
    transform = transforms.ToTensor()

    # main
    for folder in tqdm(sorted(os.listdir(args.input))):
        # mkdirs
        os.mkdir(os.path.join(args.clip_output, folder))
        os.mkdir(os.path.join(args.open_clip_output, folder))

        for img_name in os.listdir(os.path.join(args.input, folder)):
            img = Image.open(os.path.join(args.input, str(folder),
                                          img_name)).convert('RGB')
            tensor = torch.unsqueeze(transform(img), 0).cuda()

            # get embeddings
            clip_embedding = clip_embedding_extractor(tensor)
            open_clip_embedding = open_clip_embedding_extractor(tensor)

            # save
            np.save(os.path.join(args.clip_output, folder,
                                 img_name.split('.')[0]),
                    clip_embedding.detach().cpu().numpy())
            np.save(os.path.join(args.open_clip_output, folder,
                                 img_name.split('.')[0]),
                    open_clip_embedding.detach().cpu().numpy())


if __name__ == '__main__':
    main()

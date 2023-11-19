import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

import torch
import numpy as np
from argparse import ArgumentParser

def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--flower_species_list', type=str, default='flowers.txt')
    parser.add_argument('--img_dir', type=str, default='flower_imgs')
    return parser.parse_args()

def read_flower_species_list(path: str) -> list[str]:
    with open(path, mode='r') as f:
        return [line.strip().lower() for line in f]

@st.cache_resource
def load_model() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def find_k_nearest_from_text(query: Image.Image, candidates: list[str], k: int) -> list[str]:
    model, processor = load_model()
    with torch.no_grad():
        output = model(**processor(
            text=candidates,
            images=[query],
            return_tensors="pt",
            padding=True
        ))
    probability = output.logits_per_image.softmax(dim=-1)
    _, topk_idxs = probability.topk(k, dim=-1)
    return np.take(candidates, np.ravel(topk_idxs.numpy()))

def find_k_nearest_from_img(query: Image.Image, candidates: dict[str, Image.Image], k: int) -> list[str]:
    model, processor = load_model()
    with torch.no_grad():
        output = model(**processor(
            text=['UNUSED'],
            images=[query] + list(candidates.values()),
            return_tensors="pt",
            padding=True
        ))
    similarity = torch.matmul(output.image_embeds, output.image_embeds.T)
    _, topk_idxs = similarity.topk(k + 1, dim=-1)
    return np.take(['DUMMY'] + list(candidates.keys()), np.ravel(topk_idxs[0].numpy()))[1:]

if __name__ == '__main__':
    args = parse_args()

    st.title('Flower Identifier')
    uploaded = st.file_uploader(
        label='Upload a flower image',
        type=['png', 'jpg', 'jpeg'],

    )
    if uploaded:
        st.image(uploaded, width=300, caption='Uploaded Image')
        top, *nearests = find_k_nearest_from_text(
            query=Image.open(uploaded),
            candidates=read_flower_species_list(args.flower_species_list),
            k=4
        )
        st.header('Search by text:')
        with st.container():
            st.subheader(f'Identified as: {top}')
            st.image(f'{args.img_dir}/{top}.jpg', width=300, caption=f'Top Match: {top}')
            st.subheader(f'Nearest 3 species are: {nearests}')
            for i, nearest in enumerate(nearests):
                st.image(f'{args.img_dir}/{nearest}.jpg', width=300, caption=f'Nearest Match: {nearest}')

        top, *nearests = find_k_nearest_from_img(
            query=Image.open(uploaded),
            candidates={s: Image.open(f'{args.img_dir}/{s}.jpg') for s in read_flower_species_list(args.flower_species_list)},
            k=4
        )
        st.header('Search by image:')
        with st.container():
            st.subheader(f'Identified as: {top}')
            st.image(f'{args.img_dir}/{top}.jpg', width=300, caption=f'Top Match: {top}')
            st.subheader(f'Nearest 3 species are: {nearests}')
            for i, nearest in enumerate(nearests):
                st.image(f'{args.img_dir}/{nearest}.jpg', width=300, caption=f'Nearest Match: {nearest}')

import sys
import os
from txt2video import *
import time
from sd_video_utils import *
from torch import autocast

image_args = ImageArgs()
path_args = PathArgs()

model_state = load_model(path_args, optimized=True)
path_args.rife_path = './ECCV2022-RIFE/train_log'
time.sleep(1)

image_args.steps = 36
image_args.scale = 10

image_args.W = 576
image_args.H = 576

root_path = 'outputs/nice_images'
prompts = [
    #"a room full of servers, datacenters, modern looking, AI datacenters, detailed photography, realistic, 4k",
    #"a nice city skyline, futuristic london skyline, skyscrapers, modern, color photography, 4k, ecobrutalism, trees",
    #"a futuristic image, a mind imagining the future of AI, deep, detailed, futuristic architecture and design, 4k artstation"
    "people in front of a screen, a futuristic computer interface, holographic, people interacting with it, clean futuristic design, artstation 4k detailed, modern",
    "colorful rainbow AI brain, dark background, transparent, intelligence, human brain, concept art, nice design, trending on artstation"
]

precision_scope = autocast
with torch.no_grad():
    with precision_scope("cuda"):
        model_state.sampler = KDiffusionSampler(model_state.model, 'dpm_2')
        move_FS_UN_to_gpu(model_state)

        for prompt in prompts:
            C, uc = generate_embeddings([prompt], model_state)
            c = C[0]
            image_path = os.path.join(root_path, prompt.replace(' ', '_'))
            os.makedirs(image_path, exist_ok=True)
            for i in range(6):
                sample = generate_image(c=c, x=None, uc=uc, ia=image_args, ms=model_state, t_enc=None, batch_size=1, decode=True)
                image_name = f'{i}.png'
                save_image(sample, os.path.join(image_path, image_name), model_state, True)

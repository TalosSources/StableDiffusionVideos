import os
import sys
sys.path.append('custom_scripts/')
from txt2video import load_model, generate_video, generate_walk_video, ImageArgs, VideoArgs, PathArgs, FloatWrapper


path_args = PathArgs()
image_args = ImageArgs()
video_args = VideoArgs()

# Load model
path_args.image_path = './outputs/frames'
path_args.video_path = './outputs/videos'
path_args.ckpt_path = './model_weights/stable-diffusion/v_1-5.ckpt'
model = load_model(path_args, optimized=True)

# Define Vid Parameters

    #prompt
video_args.prompts = [
    "kandinsky paiting about marxism",
    ]
video_args.video_name = str(video_args.prompts[0]).replace(" ", "_") + '.mp4' 

    #performance/quality
image_args.steps = 50
image_args.W = 512
image_args.H = 512


    # animation properties
walk = False
no_noises = 2 # only used if walk = True
video_args.x = 0
video_args.y = 0
video_args.zoom = 1.01
video_args.angle = 0
video_args.frames = 30
video_args.fps = 24
video_args.strength = 0.5



# post processing
video_args.interp_exp = 2
video_args.upscale = True


if walk:
    generate_walk_video(image_args, video_args, path_args, model, no_noises, None)
else:
    generate_video(image_args, video_args, path_args, model, None)
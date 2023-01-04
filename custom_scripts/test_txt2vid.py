import sys
import os
from txt2video import *
from sd_video_utils import *
sys.path.append('stable-diffusion-2/optimizedSD')
sys.path.append('ECCV2022-RIFE')

#Placeholder "main" code
cfg_path = './stable-diffusion-2/configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = './stable-diffusion-2/optimizedSD/v1-inference.yaml'
ckpt_path = './stable-diffusion-2/models/ldm/stable-diffusion-v1/model.ckpt'

image_args = ImageArgs()
video_args = VideoArgs()
path_args = PathArgs()
video_path = 'outputs/videos'

model_state = load_model(optimized_cfg_path, ckpt_path, optimized=True)

video_args.strength = 0.40
image_args.steps = 40
image_args.W = 768
image_args.H = 512
image_args.scale = 10
video_args.zoom = 1.03
video_args.x = -3.0
video_args.y = -2.0
video_args.upscale = True
video_args.color_match = True

video_args.prompts = ["a surrealist cubist painting about a child dream, cartoony sun and moon smiling, comforting colors, 19th century child art vibe, artstation 4k"]

video_args.fps = 30
video_args.frames = 50
video_args.interp_exp = 2
video_args.video_name = "child_dream_4.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_video(image_args, video_args, path_args, model_state)


#video_args.correct = False
#video_args.video_name = "forest_autumn_uncorrected.mp4"
#path_args.video_path = os.path.join(video_path, video_args.video_name)
#generate_video(image_args, video_args, path_args, model_state)


#video_args.seed = 57194
#video_args.correct = True
#video_args.video_name = "forest_autumn_corrected_2.mp4"
#path_args.video_path = os.path.join(video_path, video_args.video_name)
#generate_video(image_args, video_args, path_args, model_state)


#video_args.correct = False
#video_args.video_name = "forest_autumn_uncorrected_2.mp4"
#path_args.video_path = os.path.join(video_path, video_args.video_name)
#generate_video(image_args, vid
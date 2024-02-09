import sys
import os
from txt2video import *
from sd_video_utils import *

image_args = ImageArgs()
video_args = VideoArgs()
path_args = PathArgs()
path_args.video_path = 'outputs/videos'

model_state = load_model(path_args, optimized=True)

image_args.steps = 40
image_args.W = 448
image_args.H = 448
image_args.scale = 10
video_args.zoom = 1
video_args.x = 10
video_args.y = 0
video_args.angle = 0
video_args.color_match = True

video_args.strength = 0.4

solarpunk = "a solarpunk futuristic utopian version of osaka japan, lively streets, organic nature, stunning digital art by thomas kinkade, incorporates cultural tradition and architecture, light, nature, forest, city, utopian"

japan_spaceship = "an alien spaceship in japanese traditional artstyle, vibrant colors, 4k, greg rutkowski alphonse mucha surrealist colors"
wave = "the wave by hokusai in space, in the style of cyril rolando,wave depicted as a galaxy, colorful vibrant digital art psychedelic surrealist artstation 4k"
steampunk = "a huge vertical steampunk fantasy city by marc simonetti, waterfalls, mountains and big traditional architecture buildings, artstation 4k"
trippy = "psychedelic fractal surrealist digital art by alan lee"
trippy2 = "psychedelic fractal surrealist digital art by john berkey kandinsky"
cyberpunk = "cyberpunk surrealist megacity in an alien planet digital art artstation john berkey"
time = "time dimension artwork 4th dimension trippy artstation by donato giancola"

india = "traditional indian mythology artwork, indian style, artstation 4k"
beach = "a beautiful artwork about a beach, modern digital art painting, cute, heart warming"
beach2 = "A beautiful painting of fort lauderdale florida by greg rutkowski and thomas kinkade, sunny day, beach with blue clear ocean water, trending on artstation, cinematic, wallpaper, 8k, ultra-detailed, high resolution, artstation, award-winning, cinematic lighting, lumen global illumination"


video_args.prompts = [beach2]


video_args.fps = 20
video_args.frames = 15
video_args.interp_exp = 3

path_args.rife_path = './ECCV2022-RIFE/train_log'

video_args.seed = 7777

video_args.video_name = "not_upscaled.mp4"
video_args.upscale = False
generate_video(image_args, video_args, path_args, model_state)

video_args.video_name = "upscaled.mp4"
video_args.upscale = True
generate_video(image_args, video_args, path_args, model_state)

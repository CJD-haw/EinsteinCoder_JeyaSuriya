import os
from moviepy.editor import *
from PIL import Image

ABSOLUTE_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(ABSOLUTE_PATH)

FRAMES_DIR = os.path.join(BASE_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

SOURCE_PATH = os.path.join(BASE_DIR, "sample.mp4")

clip = VideoFileClip(SOURCE_PATH)

(nframes, fps) = (clip.reader.nframes, clip.reader.fps)
length = int(clip.duration) + 1

print(nframes, fps, length)

for fcount, img_frame in enumerate(clip.iter_frames()):
    
    img_name = f"{fcount}.jpg"
    img_path = os.path.join(FRAMES_DIR, img_name)
    
    img_source = Image.fromarray(img_frame)
    img_source.save(img_path)




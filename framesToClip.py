import os
from moviepy.editor import *
from PIL import Image

ABSOLUTE_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(ABSOLUTE_PATH)

FRAMES_DIR = os.path.join(BASE_DIR, "frames")
CLIP_DIR = os.path.join(BASE_DIR, "clip")
os.makedirs(CLIP_DIR, exist_ok=True)

DESTINATION_PATH = os.path.join(CLIP_DIR, "sample.mp4")
    
LIST_DIR = os.listdir(FRAMES_DIR)

# FILE_PATH = [os.path.join(FRAMES_DIR, fname) for fname in LIST_DIR if fname.endswith("jpg")]  

print(LIST_DIR)

"""
FILE_PATH = []    

for fname in LIST_DIR:
    if fname.endswith("jpg"):
        path = os.path.join(FRAMES_DIR, fname)
        FILE_PATH.append(path)
"""

FILE_PATH = {}

for root, dirs, files in os.walk(FRAMES_DIR):
    for fname in files:
        filepath = os.path.join(root, fname)
        try:
            key = float(fname.replace(".jpg", " "))
        except:
            key = None
        if key != None:
            FILE_PATH[key] = filepath

for k in sorted(FILE_PATH.keys()):
    print(k)
    
# clip = ImageSequenceClip(FILE_PATH, fps=20)
# clip.write_videofile(DESTINATION_PATH)

import imageio
import os

from glob import glob

data_dir = '/home/tommy8054/MovieQA_benchmark/story/video_clips'
DIR_PATTERN_ = 'tt*'
VIDEO_PATTER_ = '*.mp4'

videos_dirs = [d for d in glob(os.path.join(data_dir, DIR_PATTERN_)) if os.path.isdir(d) ]

def exist_make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

videos_clips = {}
for d in videos_dirs:
    basename = d.split('/')[-1]
    videos_clips[basename] = glob(os.path.join(d, VIDEO_PATTER_))
    print(len(videos_clips[basename]))

for key in videos_clips.keys():
    for video in videos_clips[key]:
        basename = video[:-4]
        exist_make_dirs(basename)
        try:
            reader = imageio.get_reader(video)

        except:
            pass
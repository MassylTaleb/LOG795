import sys

from pre_processing.extract_features import get_frames_shot
from pre_processing.ffprob_shot_segmentation import ffprobe_shot_segmentation
from pre_processing.load_video import load_video
from summarization.demo import demo

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = '../data/video_to_summarize.mp4'
        load_video(sys.argv[1])
        ffprobe_shot_segmentation(path)
        get_frames_shot(path)
        demo(video_name=path)
    else:
        raise ValueError("A youtube link in the arguments is necessary to run script.")

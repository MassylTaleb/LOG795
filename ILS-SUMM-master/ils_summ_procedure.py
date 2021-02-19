#!/usr/bin/env python
import os
import sys

import pre_processing
import summarization


if __name__ == '__main__':
    if len(sys.argv) == 2:
        video_name = 'video_to_summarize.mp4'
        path = os.path.join(os.getcwd(), "data", video_name)
        pre_processing.load_video(sys.argv[1], path)
        pre_processing.ffprobe_shot_segmentation(video_name)
        pre_processing.get_frames_shot(path)
        summarization.video_summarize(path)
    else:
        raise ValueError("A youtube link in the arguments is necessary to run script.")

#!/usr/bin/env python
import os
import sys

import pre_processing
import summarization


if __name__ == '__main__':
    if len(sys.argv) == 3:
        video_name = 'video_to_summarize.mp4'
        path = os.path.join(os.getcwd(), "data", video_name)
        pre_processing.load_video(sys.argv[1], path)
        pre_processing.ffprobe_shot_segmentation(video_name)
        pre_processing.get_frames_shot(path)
        summarization.video_summarize(path, sys.argv[2])
    if len(sys.argv) == 2:
        raise ValueError("A summarization ratio is necessary in the arguments to create the trailer. "
                         "The value has to be between 0 and 1.")
    if len(sys.argv) == 1:
        raise ValueError("A youtube link in the arguments is necessary to run script.")

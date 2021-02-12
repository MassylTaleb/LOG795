import sys
import os


def load_video(link_video):
    if os.path.exists('../data/video_to_summarize.mp4'):
        os.remove('../data/video_to_summarize.mp4')
    os.system('cd ../data && youtube-dl -o video_to_summarize.mp4 "{}"'.format(link_video))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        load_video(sys.argv[1])
    else:
        raise ValueError("A youtube link in the arguments is necessary to run script.")

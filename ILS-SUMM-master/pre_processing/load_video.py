import sys
import os


def load_video(link_video, path):
    if os.path.exists(path):
        os.remove(path)
    os.system('cd ./data && youtube-dl -f "best[ext=mp4]/best" '
              '-o "video_to_summarize.mp4" "{}"'.format(link_video))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        load_video(sys.argv[1], '../data/video_to_summarize.mp4')
    else:
        raise ValueError("A youtube link in the arguments is necessary to run script.")

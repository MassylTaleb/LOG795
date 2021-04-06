import sys
import os


def load_video(link_video, video_folder, path, titre):
        # if not os.path.exists(video_path):
        #     ydl_opts = {
        #     'outtmpl' : video_path,
        #     'format': 'best[ext=mp4]/best'
        #     }
        #     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        #         ydl.download([url])
    if not os.path.exists(path):
        # os.remove(path)
        os.system('cd '+video_folder+' && youtube-dl -f "best[ext=mp4]/best" '
              '-o "{}.mp4" "{}"'.format(titre,link_video))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        load_video(sys.argv[1], '../data/video_to_summarize.mp4')
    else:
        raise ValueError("A youtube link in the arguments is necessary to run script.")

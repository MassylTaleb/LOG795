import os
import subprocess

import cv2
import numpy as np


def ffprobe_shot_segmentation(video_name):
    shot_seg_text_file = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "ILS-SUMM-master", "data", "shot_segmentation.txt")
    if not os.path.isfile(shot_seg_text_file):
        print("Ffmpeg shot segmentation in action...")

        # video_path_in_linux_style = '/'
        # full_video_path = '/'.join([video_path_in_linux_style, video_name])
        # ouput_file = '/'.join([video_path_in_linux_style, 'shot_segmentation.txt'])

        command = 'ffprobe -show_frames -of compact=p=0 -f lavfi "movie=' + video_name + ',select=gt(scene\,.2)" > ' + shot_seg_text_file
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        proc.communicate()
        print("Finished ffmpeg shot segmentation")
    print("Reading shot seg text file")
    with open(shot_seg_text_file) as f:
        content = f.readlines()
    shot_idx = [0]
    frames_per_second = 24
    i = 0
    for line in content:
        shot_idx.append(np.int(np.round(float(line.split(sep="pkt_pts_time=")[1].split(sep="|pkt_dts")[0]) * frames_per_second)))
        i = i + 1

    # Impose a minimum (Lmin) and maximum (Lmax) shot length:
    l_min = 25
    l_max = 200
    cap = cv2.VideoCapture(video_name)
    total_num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    c = np.subtract(np.append(shot_idx[1:], total_num_of_frames), shot_idx)

    # Consolidate a short shot with the following shot:
    c_without_short_shots = []
    for i in range(len(c) - 1):
        if c[i] >= l_min:
            c_without_short_shots.append(c[i])
        else:
            c[i+1] = c[i+1] + c[i]
    if c[-1] >= l_min:
        c_without_short_shots.append(c[-1])
    else:
        c_without_short_shots[-1] += c[-1]

    # Break long shot into smaller parts:
    final_c = []
    for i in range(len(c_without_short_shots)):
        if c_without_short_shots[i] <= l_max:
            final_c.append(c_without_short_shots[i])
        else:
            devide_factor = np.int((c_without_short_shots[i] // l_max) + 1)
            length_of_each_part = c_without_short_shots[i] // devide_factor
            for j in range(devide_factor - 1):
                final_c.append(length_of_each_part)
            final_c.append(c_without_short_shots[i] - (devide_factor - 1)*length_of_each_part)

    np.save("../datasets/gt_auxiliary_scripts/final_C.npy", final_c)
    return final_c


def get_framerate(video_path):
    con = "ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 " + video_path
    proc = subprocess.Popen(con, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    framerate_string = str(proc.stdout.read())[2:-5]
    a = int(framerate_string.split('/')[0])
    b = 1
    if len(framerate_string) > 1:
        int(framerate_string.split('/')[1])
    proc.kill()
    return int(np.round(np.divide(a, b)))


def extract_shots_with_ffprobe(src_video, threshold=0):
    """
    uses ffprobe to produce a list of shot
    boundaries (in seconds)

    Args:
        src_video (string): the path to the source
            video
        threshold (float): the minimum value used
            by ffprobe to classify a shot boundary

    Returns:
        List[(float, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds) and
        their associated scores
    """
    scene_ps = subprocess.Popen(("ffprobe",
                                 "-show_frames",
                                 "-of",
                                 "compact=p=0",
                                 "-f",
                                 "lavfi",
                                 "movie=" + src_video + ",select=gt(scene\," + str(threshold) + ")"),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    output = scene_ps.stdout.read()
    boundaries = extract_boundaries_from_ffprobe_output(output)
    return boundaries


def extract_boundaries_from_ffprobe_output(output):
    """
    extracts the shot boundaries from the string output
    producted by ffprobe

    Args:
        output (string): the full output of the ffprobe
            shot detector as a single string

    Returns:
        List[(float, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds) and
        their associated scores
    """
    boundaries = []
    for line in output.split('\n')[15:-1]:
        boundary = float(line.split('|')[4].split('=')[-1])
        score = float(line.split('|')[-1].split('=')[-1])
        boundaries.append((boundary, score))
    return boundaries


if __name__ == "__main__":
    # path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd()))) + "\\data\\Peppa.mp4"
    # extract_shots_with_ffprobe("Peppa.mp4", 0.3)
    ffprobe_shot_segmentation('../data/video_to_summarize.mp4')

import os

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

from .ILS_SUMM import ILS_SUMM


def video_summarize(folder_to_save_in, video_path, summ_ratio, title):
    # Load data:
    C = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'gt_auxiliary_scripts', 'final_C.npy'))
    X = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pre_processing', 'feature_vector.npy'))
    shot_sum = C.sum()
    cum_sum = np.cumsum(C[:-1])

    # Calculate allowed budget
    budget = float(summ_ratio) * np.sum(C)

    # Use ILS_SUMM to obtain a representative subset which satisfies the knapsack constraint.
    representative_points, total_distance = ILS_SUMM(X, C, budget)

    # Display Results:
    representative_points = np.sort(representative_points)
    print("The selected shots are: " + str(representative_points))
    print("The achieved total distance is: " +str(np.round(total_distance,3)))
    u, s, vh = np.linalg.svd(X)
    plt.figure()
    point_size = np.divide(C, np.max(C)) * 100
    plt.scatter(u[:, 1], u[:, 2], s=point_size, c='lawngreen', marker='o')
    plt.scatter(u[representative_points, 1], u[representative_points, 2], s=point_size[representative_points],
                c='blue', marker='o')
    plt.title('Solution Visualization (total distance = ' + str(total_distance) + ')')
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Solution_Visualization'))

    # Generate the video summary file
    video_file_path = os.path.join(video_path)
    video_clip = VideoFileClip(video_file_path)
    shotIdx = np.concatenate(([0], np.cumsum(C[:-1])))
    frames_per_seconds = np.sum(C) / video_clip.duration
    chosen_shots_clips = []
    all_start_time_clips = []
    all_end_time_clips = []
    dict = {}

    for i in range(len(representative_points)):
        curr_start_time = shotIdx[representative_points[i]] / frames_per_seconds  # [Sec]
        all_start_time_clips.append(curr_start_time)
        if representative_points[i] == (shotIdx.__len__() - 1):
            curr_end_time = video_clip.duration
        else:
            curr_end_time = (shotIdx[representative_points[i] + 1] - 1) / frames_per_seconds  # [Sec]

        all_end_time_clips.append(curr_end_time)
        chosen_shots_clips.append(VideoFileClip(video_file_path).subclip(curr_start_time, curr_end_time))
    keys = range(len(all_start_time_clips))

    x = 1
    real_clip_duration = 0
    diff_time = 0
    for i in keys:
        real_clip_duration = all_end_time_clips[i] - all_start_time_clips[i]
        clip_duration = int(real_clip_duration)
        diff_time += real_clip_duration - clip_duration
        if diff_time >= 1:
            clip_duration += 1
            diff_time -= 1
        dict[x] = all_start_time_clips[i]
        for j in range(clip_duration - 1):
            time = j + 1
            dict[x + time] = all_start_time_clips[i] + time
        x = x + clip_duration
    if not chosen_shots_clips:
        print("The length of the shortest shots exceeds the allotted summarization time")
    else:
        summ_clip = concatenate_videoclips(chosen_shots_clips)
        summ_clip.write_videofile(os.path.join(folder_to_save_in, title))

    return dict;


if __name__ == "__main__":
<<<<<<< HEAD:ILS-SUMM-master/summarization/video_summarize.py
    video_summarize('../data/video_to_summarize.mp4')
=======
    video_path
    demo()
>>>>>>> main:ILS-SUMM-master/demo.py

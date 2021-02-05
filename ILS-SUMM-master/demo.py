import numpy as np
import os
import sys
from ILS_SUMM import ILS_SUMM
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, concatenate_videoclips

def demo(video_name='Cosmus_Laundromat.mp4', summ_ratio=0.1):
    SUMM_RATIO = 0.1  # The maximum allowed ratio between the summary video and the full video.
    VIDEO_NAME = 'Cosmos_Laundromat.mp4'

    # Load data:
    X = np.load(os.path.join('data', 'shots_features.npy'))  # Load n x d feature matrix. n -> number of shots, d -> feature dimension.
    C = np.load(os.path.join('data', 'shots_durations.npy'))  # Load n x 1 shots duration array (number of frames per shot).
    shot_sum = C.sum()

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
    plt.savefig(os.path.join('data', 'Solution_Visualization'))

    # Generate the video summary file
    video_file_path = os.path.join('data', video_name)
    video_clip = VideoFileClip(video_file_path)
    shotIdx = np.concatenate(([0], np.cumsum(C[:-1])))
    frames_per_seconds = np.sum(C)/ video_clip.duration
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
    for i in keys:
        clip_duration = int(all_end_time_clips[i] - all_start_time_clips[i])
        dict[x] = all_start_time_clips[i]
        for j in range(clip_duration):
            time = j + 1
            dict[x + time] = all_start_time_clips[i] + time
        x = x + clip_duration + 1
    if chosen_shots_clips == []:
        print("The length of the shortest shots exceeds the allotted summarization time")
    else:
        summ_clip = concatenate_videoclips(chosen_shots_clips)

        summ_clip.write_videofile(os.path.join('data', "video_summary.mp4"))

if __name__ == "__main__":
    demo()

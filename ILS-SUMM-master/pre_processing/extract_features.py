import os
import re

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_features(image):
    BINS_NUMBER_PER_CHANNEL = 32
    features = np.zeros((BINS_NUMBER_PER_CHANNEL * 3), dtype=float)

    r_values = image[:, :, 0].flatten()
    g_values = image[:, :, 1].flatten()
    b_values = image[:, :, 2].flatten()

    r_hist, _ = np.histogram(r_values, BINS_NUMBER_PER_CHANNEL, [0, 256])
    normalized_r_hist = r_hist / np.sum(r_hist)
    g_hist, _ = np.histogram(g_values, BINS_NUMBER_PER_CHANNEL, [0, 256])
    normalized_g_hist = g_hist / np.sum(g_hist)
    b_hist, _ = np.histogram(b_values, BINS_NUMBER_PER_CHANNEL, [0, 256])
    normalized_b_hist = b_hist / np.sum(b_hist)

    features = np.concatenate((normalized_r_hist, normalized_g_hist, normalized_b_hist))
    return features


def string_split_by_numbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]


def get_frames_shot(path):
    shot_segmentation_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "gt_auxiliary_scripts", "final_C.npy")
    c = np.load(shot_segmentation_path, allow_pickle=False)
    frame_sep = np.sum(c)
    video_clip = VideoFileClip(path)
    cap = cv2.VideoCapture(path)
    fps = np.sum(c) / video_clip.duration
    feature_vector = []
    selected_frame = 0
    for i in c:
        selected_frame += int(i / 2)
        cap.set(1, selected_frame)
        ret, frame = cap.read()
        feature_vector.append(extract_features(frame))
    feature_vector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_vector.npy")
    np.save(feature_vector_path, feature_vector)
    return feature_vector


if __name__ == "__main__":
    get_frames_shot('../data/video_to_summarize.mp4')

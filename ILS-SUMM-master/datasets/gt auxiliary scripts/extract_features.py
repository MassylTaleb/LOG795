import numpy as np
import cv2
import glob
import os
import re

from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_features(image):
    #images = [cv2.imread(file) for file in sorted(glob.glob(os.path.join(image_dir_name, "*.jpg")), key=stringSplitByNumbers)]
    BINS_NUMBER_PER_CHANNEL = 32
    #features = np.zeros((images.__len__(), BINS_NUMBER_PER_CHANNEL * 3), dtype=float)
    features = np.zeros((BINS_NUMBER_PER_CHANNEL*3), dtype=float)

    r_values = image[:,:,0].flatten()
    g_values= image[:, :, 1].flatten()
    b_values = image[:, :, 2].flatten()

    r_hist, _ = np.histogram(r_values, BINS_NUMBER_PER_CHANNEL, [0, 256])
    normalized_r_hist =  r_hist / np.sum(r_hist)
    g_hist, _ = np.histogram(g_values, BINS_NUMBER_PER_CHANNEL, [0, 256])
    normalized_g_hist =  g_hist / np.sum(g_hist)
    b_hist, _ = np.histogram(b_values, BINS_NUMBER_PER_CHANNEL, [0, 256])
    normalized_b_hist =  b_hist / np.sum(b_hist)

    features = np.concatenate((normalized_r_hist, normalized_g_hist, normalized_b_hist))
    return features

def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

def get_frames_shot():
    C = np.load("final_C.npy",allow_pickle=False)
    frame_sep = np.sum(C)
    path = "Peppa.mp4"
    video_clip = VideoFileClip(path)
    cap = cv2.VideoCapture(path)
    fps = np.sum(C) / video_clip.duration
    feature_vector = []
    selected_frame = 0
    for i in C:
        selected_frame += int(i/2)
        cap.set(1, selected_frame)
        ret, frame = cap.read()
        feature_vector.append(extract_features(frame))
    np.save("feature_vector.npy", feature_vector)
    return feature_vector
if __name__ == "__main__":
    #extract_features(os.path.join(os.path.dirname(os.getcwd()), "images"))
    get_frames_shot()
import cv2
import numpy as np
import imutils
import math


class Extractor:

    def __init__(self):
        pass
        
    def print_hello(self):
        print('hello')


    def get_times_needed(self, dict, start_time, end_time):
        frame_needed = []

        for sum_t, real_t in dict.items():
            if start_time <= real_t and real_t <= end_time:
                frame_needed.append(sum_t)

        return frame_needed;


    def get_frame_from_video(self, time, video):
        cap = cv2.VideoCapture(video)
        fps  = cap.get(cv2.CAP_PROP_FPS)
        frame_no = fps * time
        cap.set(1, frame_no-2)
        res, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


    def get_multi_frames(self, video_dict_sum , start_time, end_time, video_path):
        frame_needed = self.get_times_needed(video_dict_sum, start_time, end_time)

        frames = []
        for time in frame_needed: 
            frames.append(self.get_frame_from_video(time, video_path)) 
        
        return frames

    def get_objects_from_image(self, image):
        # Read Image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply filters
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        canny = cv2.Canny(blurred, 100, 255, 1)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        five_percent_best = math.ceil(0.05 * len(cnts))
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:five_percent_best]
        image_number = 0
        bigger_to_smaller_boxes = []
        for c in cnts:
            rect = cv2.boundingRect(c)
            bigger_to_smaller_boxes.append(rect)

        smaller_to_bigger_boxes = reversed(bigger_to_smaller_boxes)
        smaller_to_bigger_boxes = list(smaller_to_bigger_boxes)
        boxes = smaller_to_bigger_boxes.copy()
        not_in = []

        for i, s in enumerate(smaller_to_bigger_boxes):
            for b in bigger_to_smaller_boxes:
                # If top-left inner box corner is inside the bounding box
                if s[0] > b[0] and s[1] > b[1]:
                    if (b[0] + b[2]) > (s[0] + s[2]) and (b[1] + b[3]) > (s[1] + s[3]):
                        if not_in.count(s) == 0:
                            not_in.append(s)

        for b in not_in:
            boxes.remove(b)

        # Iterate thorugh contours and filter for ROI

        ROIS = []
        for i in boxes:
            x, y, w, h = i
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = original[y:y + h, x:x + w]
            ROIS.append(ROI)
            image_number += 1

        return ROIS


if __name__ == '__main__':
    image_path = '/home/ziz/school/LOG795/object_extraction/pomme.jpg'
    image = cv2.imread(image_path)
    objects = ObjectExtractor().extract(image)
    print('Ok')
    main()
    
import cv2
import numpy as np


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
        
        # image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply filters
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        canny = cv2.Canny(blurred, 100, 255, 1)
        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Iterate thorugh contours and filter for ROI
        image_number = 0
        ROIS = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            ROIS.append(ROI)
            # cv2.imwrite("/home/ziz/school/LOG795/object_extraction/test/ROI_{}.png".format(image_number), ROI)
            image_number += 1

        return ROIS


if __name__ == '__main__':
    image_path = '/home/ziz/school/LOG795/object_extraction/pomme.jpg'
    image = cv2.imread(image_path)
    objects = ObjectExtractor().extract(image)
    print('Ok')
    main()
    
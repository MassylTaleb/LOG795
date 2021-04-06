#!/usr/bin/env python
import os
import sys
import youtube_dl
import pandas as pd 
import cv2
from datetime import timedelta

from object_extraction.Extractor import Extractor
import pre_processing
import summarization


# if __name__ == '__main__':
#     if len(sys.argv) == 3:
#         video_name = 'video_to_summarize.mp4'
#         path = os.path.join(os.getcwd(), "data", video_name)
#         pre_processing.load_video(sys.argv[1], path)
#         pre_processing.ffprobe_shot_segmentation(video_name)
#         pre_processing.get_frames_shot(path)
#         summarization.video_summarize(path, sys.argv[2])
#     if len(sys.argv) == 2:
#         raise ValueError("A summarization ratio is necessary in the arguments to create the trailer. "
#                          "The value has to be between 0 and 1.")
#     if len(sys.argv) == 1:
#         raise ValueError("A youtube link in the arguments is necessary to run script.")



if __name__ == '__main__':
    # if len(sys.argv) == 3:

    videos_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'videos')
    if not os.path.exists(videos_folder):os.makedirs(videos_folder)

    videos_sum_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'videos_summary')
    if not os.path.exists(videos_sum_folder):
        os.makedirs(videos_sum_folder)

    object_extracted_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'objects_extracted')
    if not os.path.exists(object_extracted_folder):
        os.makedirs(object_extracted_folder)


    # 1. Lire le csv
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data','csv','with_pretreatment_span_3_pfe - with_pretreatment_span_3_pfe.csv')
    df = pd.read_csv(csv_path)
    print(df.head())

    extractor = Extractor()

    # 2. Recuperer les liens des video youtube.
    urls_videos = df.url_noSubs.unique()
    
    # df_frames_objects = pd.DataFrame(colums=['index_word', 'frames','objects'])

    # 3. Recuperer le videos youtube a partir du lien
    for url in df.url_noSubs.unique():
        # print(url)
               
        first_row = df.loc[df['url_noSubs'] == url].iloc[0]
        title = first_row.title

        video_path = os.path.join(videos_folder,title+'.mp4')

        pre_processing.load_video(url, videos_folder, video_path, title)

        segmentation = pre_processing.ffprobe_shot_segmentation(videos_folder, title+'.mp4')
        frames_shot = pre_processing.get_frames_shot(video_path)

        keySum_valueReal_dict = summarization.video_summarize(videos_sum_folder,video_path, 0.1,title+"_sum.mp4")



        for index, row in df.loc[df['url_noSubs'] == url].iterrows():

            object_extracted_title_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'objects_extracted',title)
            if not os.path.exists(object_extracted_title_folder):os.makedirs(object_extracted_title_folder)

            #Get start_time as total seconds
            t_i = pd.to_datetime(row.start_time)
            start_time = (timedelta(minutes = t_i.hour, seconds=t_i.minute, microseconds=t_i.second/0.00006)).total_seconds()

            #Get end_time as total seconds
            t_f = pd.to_datetime(row.end_time)
            end_time = (timedelta(minutes = t_f.hour, seconds=t_f.minute, microseconds=t_f.second/0.00006)).total_seconds()

            # times_list = extractor.get_times_needed(keySum_valueReal_dict, start_time, end_time)
            video_sum_path = os.path.join(videos_sum_folder, title+"_sum.mp4")
            frames = extractor.get_multi_frames(keySum_valueReal_dict, start_time, end_time, video_sum_path)

            frame_count = 0
            for frame in frames:

                objects = extractor.get_objects_from_image(frame)

                object_count = 0
                for roi in objects:
                    object_name = str(index)+'_'+str(frame_count)+'_'+str(object_count)+'.png'
                    t = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(object_extracted_title_folder, object_name), t)
                    object_count += 1
                    
                frame_count+=1





    # 5. Pour chaque interval correspondant au titre du csv
    #   5.1 Extraire les frames correspondant au interval
    #   5.2 Pour chaque frame
    #       5.2.1 Extraire les objects
    # 6. Ajouter au csv, les frames et les objets en correspondance avec l'interval



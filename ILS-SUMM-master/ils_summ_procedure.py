#!/usr/bin/env python
import os, sys, glob,math
import shutil

import youtube_dl
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
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
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.exists(data_folder):os.makedirs(data_folder)

    rapport_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rapport')
    if not os.path.exists(rapport_folder):os.makedirs(rapport_folder)

    csv_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'csv')
    if not os.path.exists(csv_folder):os.makedirs(csv_folder)

    videos_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'videos')
    if not os.path.exists(videos_folder):os.makedirs(videos_folder)

    videos_sum_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'videos_summary')
    if not os.path.exists(videos_sum_folder):
        os.makedirs(videos_sum_folder)

    object_extracted_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'objects_extracted')
    if not os.path.exists(object_extracted_folder):
        os.makedirs(object_extracted_folder)


    # 1. Lire le csv
    documents = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data','csv','with_pretreatment_span_3_pfe.csv')
    documents_df = pd.read_csv(documents)
    print(documents_df.head())

    extractor = Extractor()

    # 2. Recuperer les liens des video youtube.
    urls_videos = documents_df.url_noSubs.unique()

    # df_frames_objects = pd.DataFrame(colums=['index_word', 'frames','objects'])

    # Clear object extracted folders
    folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'objects_extracted')
    for filename in os.listdir(folder_name):
        file_path = os.path.join(folder_name, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # 3. Recuperer le videos youtube a partir du lien
    for url in documents_df.url_noSubs.unique():
        # print(url)

        first_row = documents_df.loc[documents_df['url_noSubs'] == url].iloc[0]
        title = first_row.title

        video_path = os.path.join(videos_folder,title+'.mp4')

        pre_processing.load_video(url, videos_folder, video_path, title)

        segmentation = pre_processing.ffprobe_shot_segmentation(videos_folder, title+'.mp4')
        frames_shot = pre_processing.get_frames_shot(video_path)

        keySum_valueReal_dict = summarization.video_summarize(videos_sum_folder,video_path, 0.1,title+"_sum.mp4")


        # 4. Extraire les objects
        for index, row in documents_df.loc[documents_df['url_noSubs'] == url].iterrows():


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


    print("\n\n------------------ CLUSTERING ---------------------")
    # 5. Clustering
    extracted_objects_path = os.path.join(object_extracted_folder,'*','*.png')
    objects = glob.glob(extracted_objects_path)
    objects_df = pd.DataFrame(objects,columns=['object_path'])
    objects_df['object_path'].astype(str)


    # images = []
    images_flat = []
    for index, row in objects_df.iterrows():
        img = cv2.imread(row.object_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img, (240,240))
        # images.append(img)
        images_flat.append(img.flatten())
        
    # images = np.array(images)
    images_flat = np.array(images_flat)


    nb_clusters = 18
    print("nb_cluster : {}".format(nb_clusters))
    
    kmeans = KMeans(n_clusters=nb_clusters, n_jobs=-1, random_state=22)
    kmeans.fit(images_flat)


    # DataFrame Building
    object_ids = []
    document_id = []
    frames = []
    objets_num = []


    dict_info = {}
    for index, row in objects_df.iterrows():
    #     print(os.path.basename(row.path[:-4]).split('_'))

        object_ids.append(os.path.basename(row.object_path[:-4]))
        info = os.path.basename(row.object_path[:-4]).split('_')
        
        document_id.append(info[0])
        frames.append(info[1])
        objets_num.append(info[2])


    objects_df['object_id'] = object_ids
    objects_df['document_id'] = document_id
    objects_df['frame_id'] = frames
    objects_df['object_nun'] = objets_num

    # Re-ordering
    objects_df = objects_df[['object_id','document_id', 'frame_id','object_nun','object_path']]

    objects_df['cluster'] = kmeans.labels_


    # Centroids 
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, images_flat)
    centroids = []
    for i in range(len(objects_df)):
        if(i in closest): centroids.append(1)
        else: centroids.append(0)
            
    objects_df['centroid'] = centroids  

    print("\n-----------------Object Cluster CSV ---------------------\n")
    print(objects_df.head(10))
    objects_df.to_csv(os.path.join(csv_folder,'objects_clusters.csv'))
    print('\n\n')
    # Assosiation

    # Creating Topic dataframe  [Topic -- Object]
    documents_df = documents_df.rename(columns={"Unnamed: 0":'document_id'})

    topics = pd.DataFrame()

    for index, row in documents_df.iterrows():
        for topic in row.sentence.split(' '):
            data = {'document_id':row.document_id,'topic':topic}
            topics = topics.append(data,ignore_index=True)

    topics['document_id'] = topics['document_id'].astype(np.int)
    objects_df['document_id'] = topics['document_id'].astype(np.int)
    topic_objet = topics
    topic_objet = topics.merge(objects_df, left_on='document_id',right_on='document_id')

    print("\n-----------------Topic Object CSV ---------------------\n")
    print(topic_objet.head(10))
    topic_objet.to_csv(os.path.join(csv_folder,'topic_object.csv'))
    print('\n\n')

    # Creating Association Matrix between [Topic -- Cluster]
    matrice_topic_objet = topic_objet.groupby(['topic','cluster']).size().unstack(fill_value=0)

    print("\n-----------------Matric Association Topic-Cluster CSV ---------------------\n")
    print(matrice_topic_objet.head(10))
    matrice_topic_objet.to_csv(os.path.join(csv_folder,'topic_cluster.csv'))
    print('\n\n')


    # Plotting result
    # -----------------------------------------------------------------------------------------
    pdf = matplotlib.backends.backend_pdf.PdfPages('D:/ETS/BAC/PFE/LOG795/ILS-SUMM-master/data/rapport/test.pdf')
    clusters = np.sort(objects_df.cluster.unique())

    for cluster in clusters:
        cl = objects_df.loc[objects_df['cluster'] == cluster]
        qty = len(cl)
        x = round(math.sqrt(qty)) +1
        y = x-1
    #     figsize=(10, 7)
        fig = plt.figure()
    #     fig.title("Cluster : {}".format(cluster))
        fig.add_subplot(x,y,1)
        fig.suptitle('Cluster : {}'.format(cluster), fontsize=16)

        i = 0
        for index, row in cl.iterrows():
            
            fig.add_subplot(x,y,i+1)
            
            img = cv2.imread(row.object_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if(row.centroid == 1):
                img = cv2.imread(row.object_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_witdh = img.shape[1]
                img_height = img.shape[0] 
                top_border = round(img_witdh*0.1)
                left_border = round(img_height*0.1)
                img = cv2.copyMakeBorder(
                    img, top_border, top_border, left_border, left_border, 
                    cv2.BORDER_CONSTANT, 
                    value=[200,0,0]
                )

            else:
                img = cv2.imread(row.object_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.axis('off')
            plt.imshow(img)
            i+=1
    
        pdf.savefig(fig)

    pdf.close()
        # plt.savefig('/home/ziz/school/LOG795/ILS-SUMM-master/data/rapport/test.pdf')


    # 5. Pour chaque interval correspondant au titre du csv
    #   5.1 Extraire les frames correspondant au interval
    #   5.2 Pour chaque frame
    #       5.2.1 Extraire les objects
    # 6. Ajouter au csv, les frames et les objets en correspondance avec l'interval



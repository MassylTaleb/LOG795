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
from Clustering.Estimator import Estimator
import pre_processing
import summarization
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns; sns.set_theme()
from Visualisation.Visualisation import Visualisation



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
    print("\n-- Getting all objects --\n")
    extracted_objects_path = os.path.join(object_extracted_folder,'*','*.png')
    objects = glob.glob(extracted_objects_path)

    objects_df =  pd.DataFrame(columns = ['object_id','document_id','frame_id','object_num','object_path'])

    for obj_path in objects:
        object_id = os.path.basename(obj_path[:-4])
        ids_info = object_id.split('_')
    #     print(obj, object_id, ids_info)
        objects_df  = objects_df.append({'object_path':obj_path,
                                        'object_id': object_id,
                                        'document_id':ids_info[0],
                                        'frame_id':ids_info[1],
                                        'object_num':ids_info[2]},ignore_index=True)
        
    objects_df = objects_df.sort_values(by=['document_id','frame_id','object_num'])
    print(objects_df.head())


    # Load piexels as flat vector into images_flat
    images_flat = []

    for obj_path in objects_df['object_path']:
        img = cv2.imread(obj_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (240,240)).flatten()
        images_flat.append(img)
        
    images_flat = np.array(images_flat)


    # Instantiate the clustering model and visualizer

    # Estimate K cluster
    model = KMeans(random_state=22)
    nb_clusters = Estimator(rapport_folder).estimate(model, k_min =1, k_max =20, values = images_flat, methode="elbow")
    
    
    nb_clusters = 10

    
    print("\n-------------- K-MEANS --------------")
    # K-MEANS Clustering
    kmeans = KMeans(n_clusters=nb_clusters, n_jobs=-1, random_state=22)
    kmeans.fit(images_flat)


    print("\n--- Object Cluster Centroid dataset --- \n")
    # Get Centroids 
    closest_index, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, images_flat)
    centroids = []
    for i in range(len(objects_df)):
        if(i in closest_index): centroids.append(1)
        else: centroids.append(0)

    objects_df['cluster'] = kmeans.labels_
    objects_df['centroid'] = centroids  

    print(objects_df.head(5))
    print("Saving csv at :{}".format(os.path.join(csv_folder,'objects_clusters.csv')))
    objects_df.to_csv(os.path.join(csv_folder,'objects_clusters.csv'))
    print('\n\n')


    # Assosiation
    print("\n-------------- ASSOCIATION --------------")
    # Creating Topic dataframe  [Topic -- Object]

    # Reading Documents_topic_dist
    doc_topic = pd.read_csv(os.path.join(csv_folder,'doc_topic_dist.csv'), index_col=0,dtype=float)
    best_topic = doc_topic.idxmax(axis=1)
    
    # Document_id - Best_Topic_id
    doc_topic = pd.DataFrame({'document_id':best_topic.index.astype(np.int),
                                'topic_id':best_topic.values})


    # Reading topic_terms_dist
    topic_terms = pd.read_csv(os.path.join(csv_folder,'topic_terms_dist.csv'), index_col=0, dtype=float)
    nb_terms_max = 10
    best_terms = topic_terms.apply(lambda s: s.abs().nlargest(nb_terms_max).index.tolist(), axis=1)

    best_terms = pd.DataFrame(data=best_terms)

    topic_terms = pd.DataFrame({'topic_id':best_terms.index.astype(np.int),
                                'terms':best_terms[0]})
    # Spliting all terms by topic_id
    topic_terms = topic_terms.assign(topic_id=topic_terms.topic_id).explode('terms')

    # Merging all terms and document ON topic_id 
    doc_topic.topic_id = doc_topic.topic_id.astype(np.int)
    topic_terms.topic_id = topic_terms.topic_id.astype(np.int)
    doc_topic_terms = doc_topic.merge(topic_terms, left_on='topic_id',right_on='topic_id')

    # Merging Doc_topic_terms And Object ON Document_id
    doc_topic_terms['document_id']= doc_topic_terms['document_id'].astype(np.int)
    objects_df['document_id'] = objects_df['document_id'].astype(np.int)
    doc_topic_terms_object = doc_topic_terms.merge(objects_df, left_on='document_id',right_on='document_id')

    # Saving CSV
    print(doc_topic_terms_object.head(5))
    print("saving csv at: {}".format(os.path.join(csv_folder,'doc_topic_terms_object.csv')))
    doc_topic_terms_object.to_csv(os.path.join(csv_folder,'doc_topic_terms_object.csv'))

 

    print("\n-------------- GENERATING VISUALIZATION --------------")
    visual = Visualisation()

    # Nombre de lien entre Term et Cluster
    matrice_terms_objet = doc_topic_terms_object.groupby(['terms','cluster']).size().unstack(fill_value=0)
    matrice_terms_objet.to_csv(os.path.join(csv_folder,'topic_cluster.csv'))

    visual.heatmap(matrice_terms_objet, 
                    title= 'Nombre de lien en le Terme et le Cluster',
                    x_label='Clusters', 
                    y_label='Terms', 
                    saving_folder_path = rapport_folder,
                    file_name='Terms_cluster')


    # Nombre de lien entre Topic et Cluster
    matrice_topic_objet = doc_topic_terms_object.groupby(['topic_id','cluster']).size().unstack(fill_value=0)
    matrice_topic_objet.to_csv(os.path.join(csv_folder,'topic_cluster.csv'))

    visual.heatmap(matrice_topic_objet, 
                title= 'Nombre de lien en le Topic et le Cluster',
                x_label='Clusters', 
                y_label='Topic', 
                saving_folder_path = rapport_folder,
                file_name='Topic_cluster')


    # Illustrating and Saving all clusters int a pdf
    visual.visualise_clusters(objects_df = objects_df, 
                              saving_folder_path = rapport_folder,
                              pdf_file_name = 'Clusters_ilustration')



import glob, os, cv2, math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


class Visualisation:

    def __init__(self):
        pass

    def plot_clusters(self, df):
        pass
    
    def heatmap(self, matrix, title, x_label, y_label, saving_folder_path, file_name):
        
        fig, ax = plt.subplots(figsize=(10,20)) 
        ax = sns.heatmap(matrix,annot=True, cmap="YlGnBu",  fmt='d', cbar=False, square=True)

        ax.set_title(title, fontsize = 20)
        ax.set_xlabel(x_label, fontsize = 15)
        ax.set_ylabel(y_label, fontsize = 15)
        
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')

        saving_path = os.path.join(saving_folder_path, file_name+'.png')
        plt.savefig(saving_path,bbox_inches='tight')
        print("Saving HeatMap - "+title+" : {} ".format(saving_path))


    def visualise_clusters(self, objects_df, saving_folder_path, pdf_file_name):

        saving_path = os.path.join(saving_folder_path, pdf_file_name +'.pdf')
        pdf = matplotlib.backends.backend_pdf.PdfPages(saving_path)

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
        plt.savefig(saving_path)
        print("Cluster visulization saved at : {}".format(saving_path))


import glob, os, cv2, math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Visualisation:

    def __init__(self):
        pass

    def plot_clusters(self, df):
        pass
    
    def heatmap(self, df):
        clusters = np.sort(df.cluster.unique())

        for cluster in clusters:
            cl = df.loc[df['cluster'] == cluster]
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
                
                img = cv2.imread(row.path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if(row.centroid == 1):
                    img = cv2.imread(row.path)
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
                    img = cv2.imread(row.path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img)
                i+=1

        plt.savefig('/home/ziz/school/LOG795/ILS-SUMM-master/data/rapport/test.pdf')
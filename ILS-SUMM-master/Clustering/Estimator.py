import os
from yellowbrick.cluster import KElbowVisualizer

class Estimator:

    def __init__(self, rapport_path):
        self.rapport_path = rapport_path


    def estimate(self, model, k_min:int, k_max:int, values:list, methode="all"):
        
        switcher = {
            'elbow' : elbow,
            'silhouette' : silhouette,
            'dandrogramme' : dandrogramme
        }

        estimator = switcher.get(methode, lambda: 'estimator methode not valid')
        return estimator(model, k_min =1, k_max =20, values = values)


    def elbow(self, model, k_min:int, k_max:int, values:list):
        print('-----Estimateur Elbow -----')
        
        visualizer = KElbowVisualizer(model, k=(k_min, k_max), timings=True)
        visualizer.fit(values)
        saving_path = os.path.join(self.rapport_path, "elbow.png")
        visualizer.show(saving_path)

        K = visualizer.elbow_value_
        
        print('Visual graph saved at : {}'.format(saving_path))
        print('Best K : {}'.format(K))
        print("---------------------------")

        return K

    def silhouette(self, model, k_min:int, k_max:int, values:list):
        print('-----Estimateur Elbow -----')
        print('-----A FAIRE -----')


    def dandrogramme(self, model, k_min:int, k_max:int, values:list):
        print('-----Estimateur Silhouette -----')
        print('-----A FAIRE -----')



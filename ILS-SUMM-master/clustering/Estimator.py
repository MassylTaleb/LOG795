

class Estimator:

    def __init__(self, rapport_path):
        self.rapport_path = rapport_path


    def elbow(self, model, k_min:int, k_max:int, values:list):
        
        visualizer = KElbowVisualizer(model, k=(k_min, k_max), timings=True)
        visualizer.fit(values)
        saving_path = os.path.join(self.rapport_path, "elbow.png")
        visualizer.show(saving_path)

        K = visualizer.elbow_value_
        print('--------- Estimateur Elbow ----------')
        print('Visual graph saved at : {}'.format(saving_path))
        print('Resultats  K estimaté : {}'.format(K))
        print("-------------------------------------")

        return K

    def 

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class Modeler:
    def __init__(self):
        self.df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
        try: self.model = joblib.load('iris.model')
        except: self.model = None

    def fit(self):
        print('Entrenando el Modelo')
        if not os.path.exists('iris.model'):
            x = self.df.drop('species', axis=1)
            y = self.df['species']
            self.model = DecisionTreeClassifier().fit(x, y)
            print('Persistiendo el Modelo')
            joblib.dump(self.model, 'iris.model')

    def predict(self, measurement):
        print(len(measurement))
        if not os.path.exists('iris.model'):
            raise Exception('Por favor, entrene el modelo...') 
        #if len(measurement) == 4:
        #    print(len(measurement))
        #    raise Exception('Por favor, rectifique los parámetros de entrada.') 

        prediction = self.model.predict([measurement])
        return prediction[0]

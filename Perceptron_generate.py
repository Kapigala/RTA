import numpy as np
import pandas as pd
import sklearn.datasets as sk
import pickle



class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        print(self.w_[1:],self.w_[0])
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)

#Load dataset
iris = sk.load_iris()
df_full = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                       columns= iris['feature_names'] + ['species'])
df = df_full[(df_full['species'] != 2)]

#Ensure float64
X=df.iloc[:,0:4].astype("float64")
y=df['species'].astype("float64")

#Fit model
model=Perceptron(n_iter=100)
model.fit(X.to_numpy(),y.to_numpy())

#Save Pickle
o_file=open('perceptron_model.pkl','wb')
pickle.dump(model,o_file)
o_file.close()

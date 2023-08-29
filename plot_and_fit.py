import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

class Postprocessing:
    def __init__(self, filename= 'ALL.xlsx'):
        """
        Parameters:
        filename (str): The path of the Excel file.
        """
        self.filename = filename
        self.df = pd.read_excel(filename)
        self.model = None

    def plot_data(self):
        """
        Plot a 3D scatter plot.
        """
        width = self.df.iloc[:, 0]
        distance = self.df.iloc[:, 1]
        angle = self.df.iloc[:, 2]
        force = self.df.iloc[:, 6]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(width, angle, distance, c=force, cmap='jet')

        ax.set_xlabel('ds (µm)')
        ax.set_zlabel('dist (µm)')
        ax.set_ylabel('theta (deg)')

        plt.colorbar(sc, pad=0.1)

        plt.show()

    def fit_data(self):
        """
        Fit the data using three models: LinearRegression, Ridge, Lasso,
        and select the best model using cross-validation.
        """
        X = self.df.iloc[:, [0, 1, 2]]
        y = self.df.iloc[:, 6]

        models = [
            ('LinearRegression', LinearRegression()),
            ('Ridge', Ridge()),
            ('Lasso', Lasso())
        ]

        results = []
        for name, model in models:
            scores = cross_val_score(model, X, y, cv=20, scoring='neg_mean_squared_error')
            results.append((name, np.mean(scores)))

        for name, score in results:
            print(f'{name}: {score}')

        best_model = max(results, key=lambda x: x[1])
        print(f'Best model: {best_model[0]}')

        self.model = dict(models)[best_model[0]]
        self.model.fit(X, y)

        coefficients = self.model.coef_
        intercept = self.model.intercept_
        print(f'Expression: force = {intercept} + {coefficients[0]}*ds + {coefficients[1]}*dist + {coefficients[2]}*theta')

    def predict(self, X_new):
        """
        Predict using the fitted model.
        Parameters:
        X_new (DataFrame): The new set of parameters.
        Returns:
        ndarray: The predicted force.
        """
        if self.model is None:
            print('Model is not fitted yet.')
            return None

        y_pred = self.model.predict(X_new)
        print(f"Input for prediction: ds = {X_new['ds (µm)'][0]} (µm), dist = {X_new['dist (µm)'][0]} (µm), theta = {X_new['theta (deg)'][0]} (deg)")
        print('Predicted force:', y_pred, 'mN')

        return y_pred

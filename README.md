# Force Prediction Model

Postprocessing for parametric scan data: 
  1. Plot 3D scatter.
  2. 2. Fit data and get function.

The model uses three parameters: width, distance, and angle, to predict the force. The data is read from an Excel file, and the model is trained using three different regression models: Linear Regression, Ridge Regression, and Lasso Regression. The best model is selected using cross-validation.

## Installation

To run the code, you need to have Python installed on your machine. 


## Usage

The `Postprocessing` class in `plot_and_fit.py` file contains the main code. You can use this class to plot the data, fit the model, and make predictions based on given width, distance, and angle.

Here is an example of how to use the `MyModel` class:

```python
import pandas as pd
from plot_and_fit import Postprocessing

filename = 'ALL.xlsx'

model = Postprocessing(filename)
model.plot_data()
model.fit_data()

X_new = pd.DataFrame([[100, 200, 0]], columns=['ds (µm)', 'dist (µm)', 'theta (deg)'])
model.predict(X_new)


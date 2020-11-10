import pickle
import math
import json
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from config import Config

# Reading in our data
x_test = pd.read_csv(str(Config.features_path / 'test_features.csv'))
y_test = pd.read_csv(str(Config.features_path / 'test_target.csv'))

# Loading in our model
model = pickle.load(open(str(Config.models_path / 'model.pickle'), 'rb'))

# Performing predictions
y_pred = model.predict(x_test)

# Calculating metrics for the model
r_squared = r2_score(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# Saving the metrics in a json file
with open(str(Config.metrics_file_path),'w') as outfile:
    json.dump(dict(r_squared=r_squared, rmse=rmse), outfile)
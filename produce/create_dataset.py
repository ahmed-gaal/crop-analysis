import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

# Generating random seed
np.random.seed(Config.random_state)
random_state = Config.random_state

# Creating directory for storing our datasets
Config.original_data_path.parent.mkdir(parents=True, exist_ok=True)
Config.dataset_path.mkdir(parents=True, exist_ok=True)

# Downloading our dataset from Google Drive
gdown.download(
    'https://drive.google.com/uc?id=1q7c0UqDGMzYI67kzbpq9xAfBXHMkR_O_',
    str(Config.original_data_path)
)
# Reading our data into a pandas Dataframe
df = pd.read_csv(str(Config.original_data_path), encoding='latin1')

# Split data into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2,
                                     random_state=random_state)

# Saving our training and testing data into our directory
df_train.to_csv(str(Config.dataset_path / 'train.csv'), index=None)
df_test.to_csv(str(Config.dataset_path / 'test.csv'), index=None)

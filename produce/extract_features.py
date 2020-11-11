import pandas as pd
from config import Config

# Create a directory to save our features
Config.features_path.mkdir(parents=True, exist_ok=True)

# Reading in data into a pandas Dataframe
train_df = pd.read_csv(str(Config.dataset_path / 'train.csv'))
test_df = pd.read_csv(str(Config.dataset_path / 'test.csv'))

# Creating a function to extract features


def feature_extraction(df):
    return df[['Area harvested','Yield','Years']]


train_features = feature_extraction(train_df)
test_features = feature_extraction(test_df)

# Saving the features extracted from above to our directory
train_features.to_csv(str(Config.features_path / 'train_features.csv'), index=None)
test_features.to_csv(str(Config.features_path / 'test_features.csv'), index=None)

train_df.Production.to_csv(str(Config.features_path / 'train_target.csv'), index=None)
test_df.Production.to_csv(str(Config.features_path / 'test_target.csv'), index=None)
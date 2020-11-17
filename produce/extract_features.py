import pandas as pd
from config import Config
from sklearn.preprocessing import StandardScaler
# Create a directory to save our features
Config.features_path.mkdir(parents=True, exist_ok=True)

# Reading in data into a pandas Dataframe
train_df = pd.read_csv(str(Config.dataset_path / 'train.csv'))
test_df = pd.read_csv(str(Config.dataset_path / 'test.csv'))

# Creating a function to extract and preprocess features


def feature_extraction(df):
    x = df[['Area harvested', 'Years']]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x


train_features = feature_extraction(train_df)
test_features = feature_extraction(test_df)

# Saving the features extracted from above to our directory
pd.DataFrame(train_features).to_csv(
    str(Config.features_path / 'train_features.csv'), index=None
    )
pd.DataFrame(test_features).to_csv(
    str(Config.features_path / 'test_features.csv'), index=None
    )

# Preprocessing the target extracted


def preprocess(df):
    x = df['Production'].values.reshape(-1, 1)
    scaler = StandardScaler()
    return scaler.fit_transform(x)


train_target = preprocess(train_df)
test_target = preprocess(test_df)


# Saving our target to a dataframe
pd.DataFrame(train_target).to_csv(
    str(Config.features_path / 'train_target.csv'), index=None
    )
pd.DataFrame(test_target).to_csv(
    str(Config.features_path / 'test_target.csv'), index=None
    )

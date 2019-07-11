import pandas as pd

class Data(object):
    
    @property
    def train_df(self):
        return pd.read_csv('MURA-v1.1/train_image_paths.csv', header=None, names=['FilePath'])

    @property
    def valid_df(self):
        return pd.read_csv('MURA-v1.1/valid_image_paths.csv', header=None, names=['FilePath'])

    @property
    def train_labels_data(self):
        return pd.read_csv('MURA-v1.1/train_labeled_studies.csv', names=['FilePath', 'Labels'])

    @property
    def valid_labels_data(self):
        return pd.read_csv('MURA-v1.1/valid_labeled_studies.csv', names=['FilePath', 'Labels'])

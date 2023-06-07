from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd

from datasets import create_datalists, create_datasets, CT_Dataset

if __name__ == '__main__':

    imgs_list, label_list = create_datalists()

    train_data = CT_Dataset(imgs_list, label_list, split="validation")

    data_list = []
    target_list = []

    # Iterate over the dataset and extract the data
    for data, target in train_data:
        data_list.append(data.numpy())  # Convert data tensor to numpy array
        target_list.append(target.item())

    data_dict = {'data': data_list}

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_dict)

    print(df.head())

    pca = PCA(n_components=2)
    df = pca.fit_transform(df)
    print(data_list)

    # plt.scatter(data_list)

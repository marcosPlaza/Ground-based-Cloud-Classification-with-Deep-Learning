from DataLoader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = DataLoader()
    data.load_data("/Users/marcosplazagonzalez/Desktop/Ground-based_CloudClassification/Datasets/swimcat", 256, 3)

    for idx, cl in enumerate(np.rot90(np.unique(data.y, axis=0))):
        X_n = data.X[np.where((data.y == cl).all(axis=1))[0]]
        y_tmp = np.zeros((X_n.shape[0], len(data.class_names)))
        y_tmp[:, idx] = 1

        X_train, X_test, y_train, y_test = train_test_split(X_n, y_tmp, test_size=0.3, random_state=42)

        path = "/Users/marcosplazagonzalez/Desktop/Ground-based_CloudClassification/Datasets/swimcat_TrainTest/"

        count = 0
        for train_im in X_train:
            abs_path_train = path+"train/"+str(data.class_names[idx])+"/"+str(data.class_names[idx])+"_train_"+str(count)+".jpg".strip()
            print(abs_path_train)
            plt.imsave(abs_path_train, train_im)
            count += 1

        count = 0
        for test_im in X_test:
            abs_path_test = path+"test/"+str(data.class_names[idx])+"/"+str(data.class_names[idx])+"_test_"+str(count)+".jpg".strip()
            print(abs_path_test)
            plt.imsave(abs_path_test, test_im)
            count += 1
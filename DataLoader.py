import config
import pickle
import os
import numpy as np 
import imageio
from skimage import transform, io, exposure
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
DataLoader class

This class is used to load the data from the dataset (train or test exclusively).

We need a directory that at the same time stores the images' subdirectories.
Each of the subdirectories must have the name of the corresponding class.
"""
class DataLoader():    
    def __init__(self):
        self.image_size = None
        self.n_channels = None
        self.dataset_path = None
        self.alt_classes = None # dictionary with the alternative classes 

        self.X = None
        self.y = None
        self.class_names = None

    def load_from_file(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.image_size = data.image_size
        self.n_channels = data.n_channels
        self.dataset_path = data.dataset_path
        self.alt_classes = data.alt_classes

        self.X = data.X
        self.y = data.y
        self.class_names = data.class_names
    
    def load_data(self, dataset_path, image_size, n_channels, alt_classes=None, gaussian=True):
        # setting the class attributes
        self.image_size = image_size
        self.n_channels = n_channels
        self.dataset_path = dataset_path
        self.alt_classes = alt_classes

        N = 0
        labels = []

        # first we need to get labels and dimension of the dataset
        for dirname, _, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename == '.DS_Store': continue

                splits = dirname.split('/')
                    
                if self.alt_classes is None:  
                    class_name = splits[-1]
                else: 
                    class_name = self.alt_classes[splits[-1]]

                N += 1
                labels.append(class_name)

        self.X = np.zeros((N, self.image_size, self.image_size, self.n_channels))

        count = 0

        for dirname, _, filenames in tqdm(list(os.walk(self.dataset_path))):
            for filename in filenames:
                if filename == '.DS_Store': continue

                try:
                    path = os.path.join(dirname, filename)  
                        
                    im = imageio.imread(path)/255. # normalize the images by default
                    self.X[count, :, :] = transform.resize(im, (self.image_size, self.image_size, self.n_channels), mode='symmetric', preserve_range=True, anti_aliasing=gaussian)
                    
                    count += 1
                except:
                    print("Error loading image: ", path)
                    

        self.class_names = np.unique(labels)
        self.y = preprocessing.label_binarize(labels, classes=list(self.class_names))
        
        if config.model == 'ViT':
            tmp = np.where(self.y == 1)
            self.y = np.vstack(tmp[1])
            self.y = self.y.astype('int32')


if __name__ == "__main__":

    data = DataLoader()
    data.load_data("/Users/marcosplazagonzalez/Desktop/Ground-based_CloudClassification/Datasets/Swimcat-extend/", 227, 3)

    print("X shape: ", data.X.shape)
    print("y shape: ", data.y.shape)

    plt.imshow(data.X[0])
    plt.show()

    # todo not tested
    with open("./Data/swimcatdataset.data", 'wb') as datafile:
        pickle.dump(data, datafile, protocol=pickle.HIGHEST_PROTOCOL)

    print("\tDATA SAVED to {}\n".format("./Data/swimcatdataset.data"))

    """
    data = DataLoader()
    data.load_from_file("./Data/swimcatdataset.data")
    plt.imshow(data.X[0])
    plt.show() 
    """
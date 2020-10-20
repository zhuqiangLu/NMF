import os
import numpy as np
from PIL import Image
from preprocess import salt_and_pepper_noise, gaussian_noise
from sklearn.model_selection import train_test_split



def positive_init(dim1, dim2):
    m = np.random.normal(4, 1, (dim1, dim2))
    # normalize

    m = (m-m.min())/(m.max() - m.min())
    return m
def load_data(root='data/CroppedYaleB', reduce=4):
    """ 
    Load ORL (or Extended YaleB) dataset to numpy array.
    
    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.
        
    """ 
    images, labels = [], []
    shape = None
    for i, person in enumerate(sorted(os.listdir(root))):
        
        if not os.path.isdir(os.path.join(root, person)):
            continue
        
        for fname in os.listdir(os.path.join(root, person)):    
           
            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue
            
            if not fname.endswith('.pgm'):
                continue
                
            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') # grey image.

            # reduce computation complexity.
            #print(img.size)
            img = img.resize([s//reduce for s in img.size])
            shape = np.asarray(img).shape
            
            img = np.asarray(img).reshape((-1,1))
           
            # convert image to numpy array.
            #img = np.asarray(img).reshape((-1,1))
            
            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels, shape


def split_data(X, Y, ratio):
    return train_test_split(X.T, Y, train_size=ratio)
    

if __name__ == "__main__":
    from opts import parse_opt
    opts = vars(parse_opt())

    from functools import partial

    preprocess1 = partial(salt_and_pepper_noise, p=0.2, r=0.3)
    preprocess2 = partial(gaussian_noise, mu=64, sigma=16, k=2)
    X, Y = load_data(root=opts['root'], reduce=opts['reduce'])
    

    X_train, X_test, Y_train, Y_test = split_data(X, Y , 0.8)

    



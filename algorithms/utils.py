import os
import numpy as np
from PIL import Image
from preprocess import salt_and_pepper_noise, gaussian_noise



def load_data(root='data/CroppedYaleB', reduce=4, preprocess=None):
    """ 
    Load ORL (or Extended YaleB) dataset to numpy array.
    
    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.
        
    """ 
    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):
        
        if not os.path.isdir(os.path.join(root, person)):
            continue
        
        for fname in os.listdir(os.path.join(root, person)):    
            if i < 10:
                continue
            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue
            
            if not fname.endswith('.pgm'):
                continue
                
            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') # grey image.

            # reduce computation complexity.
            img = img.resize([s//reduce for s in img.size])
            
            if preprocess is not None:
                img.save("org.png")
                tem_shape = np.asarray(img).shape
                img = preprocess(img)
                Image.fromarray(img.reshape(tem_shape), "L").save('test.png')
            return None, None
            # TODO: preprocessing.

            # convert image to numpy array.
            #img = np.asarray(img).reshape((-1,1))
            
            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels


if __name__ == "__main__":
    from opts import parse_opt
    opts = vars(parse_opt())

    from functools import partial

    preprocess1 = partial(salt_and_pepper_noise, p=0.2, r=0.3)
    preprocess2 = partial(gaussian_noise, mu=64, sigma=16, k=2)
    X, Y = load_data(root=opts['root'], reduce=opts['reduce'], preprocess=preprocess2)
    print(X.shape, Y.shape)


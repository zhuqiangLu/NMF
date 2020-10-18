
import numpy as np
from PIL import Image


def salt_and_pepper_noise(img, p=0.2, r=0.3, salt_depth=255, pepper_depth=0):
    
    

    C = img.shape[-1]

    # get salt 

    n_pepper = int(C*p)
    n_salt = int(n_pepper*r)
   
    pepper = np.random.choice(C, n_pepper, replace=False)
    salt = np.random.choice(pepper, n_salt, replace=False)

    img[:,pepper] = pepper_depth
    img[:,salt] = salt_depth
    
    
    return img



def gaussian_noise(img, mu=0, sigma=64, k=1):


    disrt = np.random.normal(mu, sigma, img.shape) * k

    img = img + disrt  
    

    # truncate 
    img[img<0] = 0
    img[img>255] = 255


    return img.astype(np.uint8)










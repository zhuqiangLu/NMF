
import numpy as np
from PIL import Image


def salt_and_pepper_noise(img, p=0.2, r=0.3, salt_depth=255, pepper_depth=0):
    
    img = np.asarray(img).reshape((-1,1)).copy()

    C = img.shape[0]

    # get salt 

    n_pepper = int(C*p)
    n_salt = int(n_pepper*r)
   
    pepper = np.random.choice(C, n_pepper, replace=False)
    salt = np.random.choice(pepper, n_salt, replace=False)

    img[pepper, :] = pepper_depth
    img[salt, :] = salt_depth
    
    
    return img



def gaussian_noise(img, mu=0, sigma=64, k=1):

    # use numpy to get gaussian distribution
    img = np.asarray(img).reshape((-1,1)).copy()

    disrt = np.random.normal(mu, sigma, img.shape) * k
    print(disrt)
    img = img + disrt  
    
    # normalize
    # img = (img - np.min(img))/(np.max(img)-np.min(img))
    
    # img = img * 255

    # truncate 
    img[img<0] = 0
    img[img>255] = 255


    # s = r + np.random.normal(0, 64, r.shape)
    
    
    # s = s - np.full(s.shape, np.min(s))
    # s = s * 255 / np.max(s)
    # s = s.astype(np.uint8)

    return img.astype(np.uint8)










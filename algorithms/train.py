import numpy as np
from functools import partial
from PIL import Image
import json

from algorithms import NMF
from eval import RRE, MA
from utils import split_data, load_data
from opts import parse_opt
from preprocess import salt_and_pepper_noise, gaussian_noise

import matplotlib.pyplot as plt



def main():
    opts = vars(parse_opt())

    X, Y, ori_shape = load_data(root=opts['root'], reduce=opts['reduce'])
    
    #print(X_train_clean.shape)


    
    # the aviable algorithms
    if opts['NMF_OBJ'] is None:
        objs = ["L2",  "L21", "L1"]
    else:
        objs = [opts['NMF_OBJ']]
    

    for obj in objs:


        # split data by ratio
        X_train, X_test, Y_train, Y_test = split_data(X, Y, opts['split_ratio'])

        # copy data
        X_train_clean = (X_train.copy())

      

        # contaminate data
        if opts["noise"] == "salt_and_pepper":
            print("using salt and pepper noise")
            X_train_noise = salt_and_pepper_noise(X_train, p=opts["p"], r=opts["r"])

        elif opts["noise"] == "gaussian":
            print("using gaussian")
            X_train_noise = gaussian_noise(X_train, mu=opts["mu"], sigma=opts["sigma"])

        else:
            print("no noise")
            X_train_noise = X_train_clean
        

        # run multiple times
        for i in range(opts["epoch"]):
            
            # fit the data into the model
            D, R, E, rres = NMF(X_train_noise.T, X_clean = X_train_clean.T,  hidden_dim=opts["hidden_dim"], obj=obj, iters=opts['iters'], tol=opts['tol'])
            
            #recon = D.dot(R)

            # save
            if opts['save_rres']:
                with open('result/{}_{}.json'.format(obj, i), "w") as f:
                    json.dump({"rres": rres}, f, indent=4) 

            if opts['save_np']:
                np.save("npys/{}_{}_D.npy".format(obj, i),D) 
                np.save("npys/{}_{}_R.npy".format(obj, i),R) 
                np.save("npys/{}_{}_E.npy".format(obj, i),E) 
            
  
if __name__ == "__main__":
    main()


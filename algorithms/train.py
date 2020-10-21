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
    
    X_train, X_test, Y_train, Y_test = split_data(X, Y, opts['split_ratio'])
    
    objs = ["MSE", "KL", "L21", "ROBUSTL1"]
    
    X_train_clean = X_train.copy()
    if opts["noise"] == "salt_and_pepper":
        print("using salt and pepper noise")
        X_train_noise = salt_and_pepper_noise(X_train, p=opts["p"], r=opts["r"])

    elif opts["noise"] == "gaussian":
        print("using gaussian")
        X_train_noise = gaussian_noise(X_train, mu=opts["mu"], sigma=opts["sigma"], k=opts["k"])

    else:
        print("no noise")
        X_train_noise = X_train_clean
    
   
    #print(X_train_clean.shape)
    
    OBJ_RRE = dict()
    OBJ_RRE["noise"] = {
        "noise": opts["noise"],
        "p": opts['p'],
        "r": opts['r'],
        'mu': opts['mu'],
        'sigma': opts['sigma'],
        'k': opts['k'],
    }

    OBJ_RRE["settings"] = {
        "n_component": opts['hidden_dim'],
        "iters": opts["iters"] 
    }

    for obj in objs:

        RRE_list = list()
        for i in range(opts["epoch"]):
        
            D, R, E = NMF(X_train_noise.T, hidden_dim=opts["hidden_dim"], obj=obj, iters=opts['iters'], tol=opts['tol'])
            
            rre = RRE(X_train_clean.T, D.dot(R)+E)
        
            Image.fromarray(X_train_clean[15,:].reshape(ori_shape), "L").save("test_ori.png")
            Image.fromarray(X_train_noise[15,:].reshape(ori_shape), "L").save("test_noise.png")
            Image.fromarray(D.dot(R[:,15]).astype(np.uint8).reshape(ori_shape), "L").save("test_recon.png")
            
            
            RRE_list.append(rre)

        OBJ_RRE[obj] = RRE_list
    
    with open('result/result.json', "w") as f:
        json.dump(OBJ_RRE, f, indent=4)   
if __name__ == "__main__":
    main()


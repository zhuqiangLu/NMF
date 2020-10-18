import numpy as np
from functools import partial
from PIL import Image

from algorithms import NMF
from eval import RRE, MA
from utils import split_data, load_data
from opts import parse_opt
from preprocess import salt_and_pepper_noise, gaussian_noise


def main():
    opts = vars(parse_opt())

    X, Y, ori_shape = load_data(root=opts['root'], reduce=opts['reduce'])
    
    X_train, X_test, Y_train, Y_test = split_data(X, Y, opts['split_ratio'])
    
    
    
    X_train_clean = X_train.copy()
    if opts["noise"] == "salt_and_pepper":
        print("using salt and pepper noise")
        X_train_noise = salt_and_pepper_noise(X_train, p=opts["p"], r=opts["r"])

    elif opts["noise"] == "gaussian":
        print("using salt and pepper noise")
        X_train_noise = gaussian_noise(X_train, mu=opts["mu"], sigma=opts["sigma"], k=opts["k"])

    else:
        print("no noise")
        X_train_noise = X_train_clean
    
   
    print(X_train_clean.shape)
    RRE_list = list()
    for i in range(opts["epoch"]):
        D, R = NMF(X_train_clean, hidden_dim=opts["hidden_dim"], obj=opts["NMF_OBJ"])
        
        #model = sklearn.decomposition.NMF(n_components=200) # set n_components to num_classes.
        #D = model.fit_transform(X.T)
        #R = model.components_
        rre = RRE(X_train_clean, D.dot(R))
        print(X_train_clean - D.dot(R))
        RRE_list.append(rre)

        ori = X_train_clean[15,:]
        noise_ori = X_train_noise[15,:]
        Image.fromarray(ori.reshape(ori_shape), "L").save("test.png")
        Image.fromarray(noise_ori.reshape(ori_shape), "L").save("test_noise.png")

        recon = D.dot(R)[15,:]
        Image.fromarray(recon.reshape(ori_shape), "L").save("test_recon.png")


    print(RRE_list)

if __name__ == "__main__":
    main()


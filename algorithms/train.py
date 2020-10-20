import numpy as np
from functools import partial
from PIL import Image

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
    
   
    #print(X_train_clean.shape)
    RRE_list = list()
    import sklearn
    for i in range(opts["epoch"]):
    
        D, R = NMF(X_train_noise.T, hidden_dim=opts["hidden_dim"], obj=opts["NMF_OBJ"], iters=opts['iters'], tol=opts['tol'])
        # import sklearn
        # model = sklearn.decomposition.NMF(n_components=100, max_iter=500)

        # D = model.fit_transform(X_train_noise.T)
        # R = model.components_
        
        rre = RRE(X_train_clean.T, D.dot(R))
        # print(rre)


        rre_noise= RRE(X_train_clean.T, X_train_noise.T)
        rre_noise_recon= RRE(X_train_noise.T, D.dot(R))
        Image.fromarray(X_train_clean[15,:].reshape(ori_shape), "L").save("test_ori.png")
        Image.fromarray(X_train_noise[15,:].reshape(ori_shape), "L").save("test_noise.png")
        Image.fromarray(D.dot(R[:,15]).astype(np.uint8).reshape(ori_shape), "L").save("test_recon.png")
        print("ori recon rre: {} || ori noise rre {} || noise recon rre {}".format(rre, rre_noise, rre_noise_recon))
        
       
           
        RRE_list.append(rre)

        # ori = X_train_clean[15,:]
        # noise_ori = X_train_noise[15,:]
        # Image.fromarray(ori.reshape(ori_shape), "L").save("test.png")
        # Image.fromarray(noise_ori.reshape(ori_shape), "L").save("test_noise.png")

        # recon = D.dot(R)[15,:]
        # Image.fromarray(recon.reshape(ori_shape), "L").save("test_recon.png")


    print(RRE_list)
    plt.show()
if __name__ == "__main__":
    main()


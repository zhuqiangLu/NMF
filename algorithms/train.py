import numpy as np
from functools import partial
from PIL import Image

from algorithms import NMF
from eval import RRE, MA
from utils import split_data, load_data
from opts import parse_opt
from preprocess import salt_and_pepper_noise, gaussian_noise

import matplotlib.pyplot as plt


n_row,n_col=2,3
n_components=n_row*n_col
image_shape=(48,42)
def plot_gallery(title,images,n_col=n_col,n_row=n_row):
    plt.figure(figsize=(2.*n_col,2.26*n_row)) #创建图片，并指定图片大小
    plt.suptitle(title,size=18) #设置标题及字号大小
    
    for i,comp in enumerate(images):
        plt.subplot(n_row,n_col,i+1) #选择绘制的子图
        vmax=max(comp.max(),-comp.min())
        print(comp.shape)
        plt.imshow(comp.reshape(image_shape),cmap=plt.cm.gray,
                   interpolation='nearest',vmin=-vmax,vmax=vmax) #对数值归一化，并以灰度图形式显示
        plt.xticks(())
        plt.yticks(()) #去除子图的坐标轴标签
    plt.subplots_adjust(0.01,0.05,0.99,0.94,0.04,0.) #对子图位置及间隔调整


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
        D, R = NMF(X_train_noise.T, hidden_dim=opts["hidden_dim"], obj=opts["NMF_OBJ"], iters=opts['iters'])
        
        
        rre = RRE(X_train_clean.T, D.dot(R))
        # print(rre)
        
        Image.fromarray(X_train_clean[15,:].reshape(ori_shape), "L").save("test.png")
        
        Image.fromarray(D.dot(R[:,15]).astype(np.uint8).reshape(ori_shape), "L").save("test1.png")
        
        # for j in range(100):
            
        #     Image.fromarray(D[:,j].astype(np.uint8).reshape((ori_shape)), "L").save("D{}.png".format(j))
        

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


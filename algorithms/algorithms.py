import numpy as np
from utils import positive_init


def MSE(X, X_hat):
    # implementation of Mean Square Error
    e = np.linalg.norm(X-X_hat)/X.size
    return e


def KL_Divergence(X, X_hat):
    X_hat += 1e-7
    div = (X/X_hat) + 1e-7
    return np.sum(X * np.log(div) - X + X_hat)/X.size


def L21_norm(X, X_hat):
    l21 = np.sum(np.sqrt(np.sum(np.square(X-X_hat), axis=0, keepdims=True))) 
    return l21/X.size

def MUR_L21(X, D, R, steps, tol=1e-3):
    step = 0
    diff = float("Inf")
    N = X.shape[1]

    
    while step < steps and diff > tol:
        L21_bef = L21_norm(X, D.dot(R))
        
        Dig = (1/(np.sqrt(np.sum(np.square(X-D.dot(R)), axis=0, keepdims=True)))) * np.eye(N)
        D = D * (X.dot(Dig).dot(R.T))/(D.dot(R).dot(Dig).dot(R.T))+1e-7

        Dig = (1/(np.sqrt(np.sum(np.square(X-D.dot(R)), axis=0, keepdims=True)))) * np.eye(N)
        R =  R * (D.T.dot(X).dot(Dig))/(D.T.dot(D).dot(R).dot(Dig))+1e-7

        diff = L21_bef - L21_norm(X, D.dot(R))
        print(step, L21_norm(X, D.dot(R)), diff)
        step+=1
    return D, R


def MUR_MSE(X, D, R, steps=50, tol=1e-3):
    step = 0
    diff = float("Inf")
    
    while step < steps and diff > tol:
        MSE_bef = MSE(X, D.dot(R)) 
        R = R * ((np.dot(D.T, X)/(np.dot(np.dot(D.T, D), R))))+1e-7
        D = D * ((np.dot(X, R.T))/(np.dot(np.dot(D, R), R.T)))+1e-7

        diff = MSE_bef - MSE(X, D.dot(R)) 
        step += 1
        print(step, MSE(X, D.dot(R)), diff)
    return D, R

def MUR_KL(X, D, R, steps=500, tol=1e-3):
    step = 0
    diff = float("Inf")
    while step < steps and diff > tol:
        
        KL_bef = KL_Divergence(X, D.dot(R)) 
        
        temp = X/((np.dot(D, R))+1e-7)
        
        R = R * np.dot(D.T, temp)/np.sum(D.T, axis=1, keepdims=True)

        temp = X/((np.dot(D, R))+1e-7)
        D = D * np.dot(temp, R.T)/np.sum(R.T, axis=0, keepdims=True)

        diff = KL_bef - KL_Divergence(X, D.dot(R)) 
        step += 1
        print(step, KL_Divergence(X, D.dot(R)), diff)
        
        
    return D, R




def NMF(X, hidden_dim, iters, tol, obj="MSE"):
    '''
        implementation of algorithm presented in 
        https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf
    '''
    C = X.shape[0]
    N = X.shape[-1]
    
    D = positive_init(C, hidden_dim)
    R = positive_init(hidden_dim, N) 
    if obj == "MSE":
        D, R = MUR_MSE(X, D, R, steps=iters, tol=tol)
    elif obj == "KL":
        D, R = MUR_KL(X, D, R, steps=iters, tol=tol)
    elif obj == "L21":
        D, R = MUR_L21(X, D, R, steps=iters, tol=tol)
    else:
        D, R = MUR_MSE(X, D, R, steps=iters, tol=tol)
    return D, R
    


    




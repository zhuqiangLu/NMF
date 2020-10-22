import numpy as np
from utils import positive_init
from eval import RRE

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

def RobustNMF(X, X_hat, E, lamb):
    # the objective function of L1 Robust NMF
    t1 = np.linalg.norm(X-X_hat-E)
    t2 = lamb * np.linalg.norm(np.sum(E, axis=1), ord=0)
    return (t1 + t2)/X.size

def MUR_L21(X, D, R, X_clean,steps, tol=1e-3):
    # updating rule for L21 nmf
    '''
        X : (C, N)
        D : (C, k)
        R : (k, N)

    '''
    step = 0
    diff = float("Inf")
    N = X.shape[1]

    rres = list()
    while step < steps and diff > tol:
        L21_bef = L21_norm(X, D.dot(R))
        
        # first update D
        Dig = (1/(np.sqrt(np.sum(np.square(X-D.dot(R)), axis=0, keepdims=True)))) * np.eye(N)
        D = D * (X.dot(Dig).dot(R.T))/(D.dot(R).dot(Dig).dot(R.T))+1e-7

        # then R
        Dig = (1/(np.sqrt(np.sum(np.square(X-D.dot(R)), axis=0, keepdims=True)))) * np.eye(N)
        R =  R * (D.T.dot(X).dot(Dig))/(D.T.dot(D).dot(R).dot(Dig))+1e-7

        # compute diff for stop
        diff = L21_bef - L21_norm(X, D.dot(R))
        if X_clean is not None:
            rres.append(RRE(X_clean, D.dot(R)))
        #print(step, L21_norm(X, D.dot(R)), diff)
        step+=1
    return D, R, rres


def MUR_L2(X, D, R, X_clean,steps=50, tol=1e-3):
    '''
        X : (C, N)
        D : (C, k)
        R : (k, N)

    '''
    step = 0
    diff = float("Inf")
    rres = list()
    while step < steps and diff > tol:
        MSE_bef = MSE(X, D.dot(R)) 
        R = R * ((np.dot(D.T, X)/(np.dot(np.dot(D.T, D), R))))+1e-7
        D = D * ((np.dot(X, R.T))/(np.dot(np.dot(D, R), R.T)))+1e-7

        diff = MSE_bef - MSE(X, D.dot(R)) 
        step += 1
        if X_clean is not None:
            rres.append(RRE(X_clean, D.dot(R)))
        #print(step, MSE(X, D.dot(R)), diff)
    return D, R, rres

def MUR_KL(X, D, R, X_clean,steps=500, tol=1e-3):
    '''
        X : (C, N)
        D : (C, k)
        R : (k, N)

    '''
    step = 0
    diff = float("Inf")
    rres = list()
    while step < steps and diff > tol:
        
        KL_bef = KL_Divergence(X, D.dot(R)) 
        
        temp = X/((np.dot(D, R))+1e-7)
        
        R = R * np.dot(D.T, temp)/np.sum(D.T, axis=1, keepdims=True)

        temp = X/((np.dot(D, R))+1e-7)
        D = D * np.dot(temp, R.T)/np.sum(R.T, axis=0, keepdims=True)

        diff = KL_bef - KL_Divergence(X, D.dot(R)) 
        step += 1
        if X_clean is not None:
            rres.append(RRE(X_clean, D.dot(R)))
        #print(step, KL_Divergence(X, D.dot(R)), diff)
        
        
    return D, R, rres

def MUR_L1_ROBUST(X, D, R, X_clean,lamb=0.05, steps=500, tol=1e-3):

    '''
        X : (C, N)
        D : (C, k)
        R : (k, N)

    '''
    rres = list()
    E = np.random.normal(0, 1, X.shape) * 40
    E[(X-E) < 0 ] = 0

    C = X.shape[0]
    N = X.shape[1]
    k = D.shape[1]
    

    step = 0
    diff = float('Inf')

    while step < steps and diff > tol:

        # obj bef 
        obj_bef = RobustNMF(X, D.dot(R), E, lamb)
        # update D
        X_hat = X-E
        D = D * ((np.dot(X_hat, R.T))/((np.dot(np.dot(D, R), R.T)+1e-7)))

        #update R
        zeros = np.zeros((1, k))
        es = np.sqrt(lamb) * np.exp(np.ones((1, C)))
        
        X_curl = np.vstack((X, np.zeros(((1,N)))))
        
        D_curl = np.vstack(((np.hstack((D, np.eye(C), -1*np.eye(C)))),\
                            np.hstack((zeros, es, es))))
        E_p = (np.abs(E) + E)/2
    
        E_n = (np.abs(E) - E)/2
        
        # prepare for the computation of R
        S = np.abs(D_curl.T.dot(D_curl))
        R_curl = np.vstack((R, E_p, E_n))
        
        # prepart for the long equation
        denom = (S.dot(R_curl))+1e-7
        mol1 = D_curl.T.dot(D_curl).dot(R_curl)
        mol2 = D_curl.T.dot(X_curl)
        
        # compute the block R
        R_curl_tmp = R_curl * (1 - (mol1 - mol2)/denom)
        R_curl = np.maximum(np.zeros_like(R_curl), R_curl_tmp)
        R = R_curl[:k, :]
        #restore E
        E = R_curl[k:k+C,:] - R_curl[k+C:, :]
        
        step += 1
        obj_aft = RobustNMF(X, D.dot(R), E, lamb)
        diff = obj_bef - obj_aft
        if X_clean is not None:
            rres.append(RRE(X_clean, D.dot(R)))
        #print(step, obj_aft, diff)


    return D, R, E, rres


def NMF(X, hidden_dim, iters, tol, obj, X_clean=None):
    '''
        implementation of algorithm presented in 
        https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf

        X: contaminated data (C, N)
        X_hat: the clean data
    '''
    C = X.shape[0]
    N = X.shape[-1]
    
    D = positive_init(C, hidden_dim)
    R = positive_init(hidden_dim, N) 
    E = np.zeros_like(X)
    if obj == "L2":
        print("OBJ: L2")
        D, R, rres = MUR_L2(X, D, R, X_clean, steps=iters, tol=tol)
    elif obj == "KL":
        print("OBJ: KL")
        D, R, rres = MUR_KL(X, D, R, X_clean, steps=iters, tol=tol)
    elif obj == "L21":
        print("OBJ: L21")
        D, R, rres= MUR_L21(X, D, R, X_clean, steps=iters, tol=tol)
    elif obj == "L1":
        print("OBJ: ROBUSTL1")
        D, R, E, rres = MUR_L1_ROBUST(X, D, R, X_clean, steps=iters, tol=tol)
    else:
        D, R, rres = MUR_L2(X, D, R, X_clean, steps=iters, tol=tol)
    return D, R, E, rres
    


    




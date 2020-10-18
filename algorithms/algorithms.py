import numpy as np



def MSE(X, X_hat):
    # implementation of Mean Square Error
    e = np.linalg.norm(X-X_hat)/X.size
    return e


def KL_Divergence(X, X_hat):

    return np.sum(X * np.log(X/X_hat) - X + X_hat)    




def MUR_MSE(X, D, R, steps=5000, tol=1e-3):
    step = 0
    

    while step < steps:

        R_tmp = R.copy()
        D_tmp = D.copy()

        R = R * ((np.dot(D.T, X)/(np.dot(np.dot(D.T, D), R))))
        D = D * ((np.dot(X, R.T))/(np.dot(np.dot(D, R), R.T)))

        e_r = MSE(R_tmp, R)
        e_d = MSE(D_tmp, D) 

        if e_r < tol and e_d < tol:
            break
        step += 1
        

    return D, R

def MUR_KL(X, D, R, steps=5000, tol=1e-3):
    step = 0
   

    while step < steps:
        R_tmp = R.copy()
        D_tmp = D.copy()
        
        temp = X/(np.dot(D, R.T))
        R = R * np.dot(D.T, temp)/np.sum(D.T, axis=1, keepdims=True)

        temp = X/(np.dot(D, R.T))
        D = D * np.dot(temp, R.T)/np.sum(R.T, axis=1, keepdims=True)

        e_r = MSE(R_tmp, R)
        e_d = MSE(D_tmp, D) 

        if e_r < tol and e_d < tol:
            break
        step += 1
    return D, R


def NMF(X, k, obj="MSE"):
    '''
        implementation of algorithm presented in 
        https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf
    '''
    M = X.shape[0]
    N = X.shape[-1]
    
    # D, R initializaiton
    D = np.random.rand((N, k)) 
    R = np.random.rand((k, M))
    


    




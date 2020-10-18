import numpy as np



def MSE(X, X_hat):
    # implementation of Mean Square Error
    e = np.linalg.norm(X-X_hat)/X.size
    return e


def KL_Divergence(X, X_hat):

    return np.sum(X * np.log(X/X_hat) - X + X_hat)    



def MUR_MSE(X, D, R, steps=50, tol=1e-3):
    step = 0
    diff = tol * 10.0

    while step < steps or diff < tol:

        R = R * ((np.dot(D.T, X)/(np.dot(np.dot(D.T, D), R))))+1e-7
        D = D * ((np.dot(X, R.T))/(np.dot(np.dot(D, R), R.T)))+1e-7

        diff = MSE(X, D.dot(R))
        step += 1
        print(step, MSE(X, D.dot(R)))
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
        print(step, e_r, e_d)
        if e_r < tol and e_d < tol:
            break
        step += 1
    return D, R




def NMF(X, hidden_dim, iters, obj="MSE"):
    '''
        implementation of algorithm presented in 
        https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf
    '''
    N = X.shape[0]
    C = X.shape[-1]
    

    if obj == "MSE":
        # D, R initializaiton
        D = np.random.rand(N, hidden_dim) *40
        R = np.random.rand(hidden_dim, C) *40
        #D, R = MUR_MSE(X, D, R)
        D, R = MUR_MSE(X, D, R, steps=iters)
        return D, R
    else:
        # D, R initializaiton
        D = np.random.rand(N, hidden_dim) *40
        R = np.random.rand(hidden_dim, C) *40
        D, R = MUR_KL(X, D, R)
        return D, R
    


    




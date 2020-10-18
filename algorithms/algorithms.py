import numpy as np



def MSE(X, X_hat):
    # implementation of Mean Square Error
    e = np.linalg.norm(X-X_hat)/X.size
    return e


def KL_Divergence(X, X_hat):

    return np.sum(X * np.log(X/X_hat) - X + X_hat)    



def MUR_MSE(X, D, R, steps=100, tol=1e-3):
    step = 0
   

    while step < steps:

        # R_tmp = R
        # D_tmp = D
        
        R = R * ((np.dot(D.T, X)/(np.dot(np.dot(D.T, D), R))))+1e-7
        #print(R.shape, (np.dot(D.T, X)).shape, ((np.dot(np.dot(D.T, D), R))).shape)
        D = D * ((np.dot(X, R.T))/(np.dot(np.dot(D, R), R.T)))+1e-7

        # #e_r = MSE(R_tmp, R)
        # e_r = np.sqrt(np.sum((R_tmp-R)**2, axis=(0,1)))/R.size
        # #e_d = MSE(D_tmp, D) 
        # e_d = np.sqrt(np.sum((D_tmp-D)**2, axis=(0,1)))/D.size
        # #print(step, e_r, e_d)
        print(step, MSE(X, D.dot(R)))
        # if e_r < tol and e_d < tol:
        #     break
        #print(step)
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
        print(step, e_r, e_d)
        if e_r < tol and e_d < tol:
            break
        step += 1
    return D, R




def NMF(X, hidden_dim, obj="MSE"):
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
        D, R = MUR_MSE(X, D, R)
        return D, R
    else:
        # D, R initializaiton
        D = np.random.rand(N, hidden_dim) *40
        R = np.random.rand(hidden_dim, C) *40
        D, R = MUR_KL(X, D, R)
        return D, R
    


    




import pandas as pd
import numpy as np
from copy import deepcopy

np.set_printoptions(suppress=True)

# Function for 2sls

def reg(X, y):
    XTX_inv = np.linalg.inv(np.dot(X.T, X))
    beta_hr = np.dot(XTX_inv, np.dot(X.T, y))
    return beta_hr

def _ivregress(X, Z, y, verbose = False):
    tmp1 = np.dot(Z, np.linalg.inv(np.dot(Z.T, Z)))
    if verbose:
        print(tmp1.shape)
    X_hat = np.dot(tmp1, np.dot(Z.T, X))
    beta_2sls = reg(X_hat, y)
    if verbose:
        print(beta_2sls)
    Pz = np.dot(tmp1, Z.T)
    if verbose:
        print(Pz.shape)

    eps = y - np.dot(X, beta_2sls)
    sigma_2 = np.dot(eps.T, eps)/X.shape[0]
    
    if verbose:
        print(sigma_2)

    Var_beta_2sls = sigma_2 * np.linalg.inv(np.dot(np.dot(X.T, Pz), X))
    if verbose:
        print(np.sqrt(np.diag(Var_beta_2sls)))
    return beta_2sls, Var_beta_2sls

def ivregress_2sls(S, y_var, regs, ev, inst, verbose=False):
    # TODO: handle the constant better - without duplicating or overwriting the data
    S['_const'] = 1
    
    # exogenous variables
    ex_vars = list(set(regs) - set(ev))
    
    z_vars = ['_const'] + inst + ex_vars
    x_vars = ['_const'] + regs
    
    if verbose:
        print('Variables in Z matrix are')
        print(z_vars)
        
        print('Variables in X matrix are')
        print(x_vars)
    
    
    Z = S[z_vars].to_numpy()
    X = S[x_vars].to_numpy()
    y = S[y_var].to_numpy()
    
    beta_2sls, Var_beta_2sls = _ivregress(X, Z, y)
        
    std_err = np.sqrt(np.diag(Var_beta_2sls))
                      
    df_out = pd.DataFrame(data = np.concatenate([beta_2sls[:, None], std_err[:, None]], axis=1)
            , columns = ['Coef.', 'Std. Err.']
            , index = x_vars)
                      
    return df_out



# Functions for ts2sls
def _ts2sls(X2, X1, Z2, Z1, y1, y_z, ev_ind, verbose=False):
    tmp_z2t2_inv = np.linalg.inv(np.dot(Z2.T, Z2))
    tmp_z2tx2 = np.dot(Z2.T, X2)
    X1_hat = np.dot(Z1, np.dot(tmp_z2t2_inv, tmp_z2tx2))
    beta_ts2sls = np.dot(np.linalg.inv(np.dot(X1_hat.T, X1_hat)), np.dot(X1_hat.T, y1))
    
    n1 = Z1.shape[0]
    n2 = Z2.shape[0]
    
    beta_1s = reg(Z2, y_z)
    pred_y_z = np.dot(Z1, beta_1s)
        
    pred_X1 = deepcopy(X1).astype('float64')
    pred_X1[:, ev_ind] = pred_y_z[:,0]
    
    k_p = pred_X1.shape[1]
    eps = y1 - np.dot(pred_X1, beta_ts2sls)
    sigma_2 = np.dot(eps.T, eps)/(n1 - k_p)
    if verbose:
        print(sigma_2)

    Var_beta_2sls = sigma_2 * np.linalg.inv(np.dot(X1_hat.T, X1_hat))
    if verbose:
        print(Var_beta_2sls)
        print(np.sqrt(np.diag(Var_beta_2sls)))
    
    k_q = Z2.shape[1]
    pred_X2 = deepcopy(X2).astype('float64')
    pred_y_z2 = np.dot(Z2, beta_1s)
    pred_X2[:, ev_ind] = pred_y_z2[:, 0]
    eps_1s = X2 - pred_X2
    sigma_nu = np.dot(eps_1s.T, eps_1s)/(n2-k_q)
    if verbose:
        print(sigma_nu)

    sigma_f = sigma_2 + n1/n2 * np.dot(beta_ts2sls.T, np.dot(sigma_nu, beta_ts2sls))
    if verbose:
        print(sigma_f)
    Var_beta_ts2sls = sigma_f * np.linalg.inv(np.dot(X1_hat.T, X1_hat))
    
    if verbose:
        print(Var_beta_ts2sls)
        print(np.sqrt(np.diag(Var_beta_ts2sls)))
    
    return beta_ts2sls, Var_beta_ts2sls

# TODO: Account for the multiple endogenous variables
def ts2sls(S1, S2, y_var, regs, ev, inst, verbose=False):
    # TODO: handle the constant better - without duplicating or overwriting the data
    S1['_const'] = 1
    S2['_const'] = 1
    
    # exogenous variables
    ex_vars = list(set(regs) - set(ev))
    
    z_vars = ['_const'] + inst + ex_vars
    x_vars = ['_const'] + regs
    
    if verbose:
        print('Variables in Z matrix are')
        print(z_vars)
        
        print('Variables in X matrix are')
        print(x_vars)
    
    
    Z2 = S2[z_vars].to_numpy()
    Z1 = S1[z_vars].to_numpy()
    X2 = S2[x_vars].to_numpy()
    X1 = S1[x_vars].to_numpy()
    y1 = S1[y_var].to_numpy()
    y_z = S2[ev].to_numpy()
    
    # TODO: find a better way to communicate the positions of endogenous variables
    ev_ind = x_vars.index(ev[0])
    
    beta_ts2sls, Var_beta_ts2sls = _ts2sls(X2, X1, Z2, Z1, y1, y_z, ev_ind)
    std_err = np.sqrt(np.diag(Var_beta_ts2sls))
                      
    df_out = pd.DataFrame(data = np.concatenate([beta_ts2sls[:, None], std_err[:, None]], axis=1)
            , columns = ['Coef.', 'Std. Err.']
            , index = x_vars)
                      
    return df_out



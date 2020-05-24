import pandas as pd
import numpy as np
from copy import deepcopy

df1 = pd.read_csv('auto_data.csv', low_memory=False)
df1['group'] = 1
df2 = deepcopy(df1)
df2['group'] = 2
df1['mpg'] = np.NaN

df1['const'] = 1
df2['const'] = 1

np.set_printoptions(suppress=True)


def reg(X, y):
    XTX_inv = np.linalg.inv(np.dot(X.T, X))
    beta_hr = np.dot(XTX_inv, np.dot(X.T, y))
    return beta_hr



def iv_regress(X, Z, y):
    tmp1 = np.dot(Z, np.linalg.inv(np.dot(Z.T, Z)))
    print(tmp1.shape)
    X_hat = np.dot(tmp1, np.dot(Z.T, X))
    beta_2sls = reg(X_hat, df1[['price']].to_numpy())
    print(beta_2sls)
    Pz = np.dot(tmp1, Z.T)
    print(Pz.shape)

    eps = y - np.dot(X, beta_2sls)
    sigma_2 = np.dot(eps.T, eps)/74
    print(sigma_2)

    Var_beta_2sls = sigma_2 * np.linalg.inv(np.dot(np.dot(X.T, Pz), X))
    print(Var_beta_2sls)
    print(np.sqrt(Var_beta_2sls[0,0]))
    print(np.sqrt(Var_beta_2sls[1,1]))
    print(np.sqrt(Var_beta_2sls[2,2]))
    return beta_2sls, Var_beta_2sls

df2['const'] = 1
X = df2[['const', 'mpg', 'weight']].to_numpy()
Z = df1[['const', 'headroom', 'weight']].to_numpy()
y = df1[['price']].to_numpy()
beta_2sls, Var_beta_2sls = iv_regress(X, Z, y)
print(Var_beta_2sls)

print(np.sqrt(np.diag(Var_beta_2sls)))


def ts2sls(X2, Z2, Z1, y1, y_z):
    tmp_z2t2_inv = np.linalg.inv(np.dot(Z2.T, Z2))
    tmp_z2tx2 = np.dot(Z2.T, X2)
    X1_hat = np.dot(Z1, np.dot(tmp_z2t2_inv, tmp_z2tx2))
    print(X1_hat.shape)
    beta_ts2sls = np.dot(np.linalg.inv(np.dot(X1_hat.T, X1_hat)), np.dot(X1_hat.T, y1))
    print(beta_ts2sls)
    
    n1 = Z1.shape[0]
    n2 = Z2.shape[0]
    
    beta_1s = reg(Z2, y_z)
    
    pred_y_z = np.dot(Z1, beta_1s)
    print(pred_y_z.shape)
    
    #pred_X1 = np.concatenate((Z1[:, 0], pred_y_z, Z1[:, 2]), axis=1)
    pred_X1 = deepcopy(Z1)
    pred_X1[:, 1] = pred_y_z[:,0]
    print(pred_X1.shape)
    
    eps = y1 - np.dot(pred_X1, beta_ts2sls)
    #eps = y1 - np.dot(X2, beta_ts2sls)
    sigma_2 = np.dot(eps.T, eps)/(n1-3)
    print(sigma_2)

    Var_beta_2sls = sigma_2 * np.linalg.inv(np.dot(X1_hat.T, X1_hat))
    print(Var_beta_2sls)
    print(np.sqrt(np.diag(Var_beta_2sls)))
    
    
    eps_1s = y_z - np.dot(Z2, beta_1s)
    sigma_nu = np.dot(eps_1s.T, eps_1s)/(n2-3)
    print(sigma_nu)
    
    sigma_f = sigma_2 + beta_2sls[1]*sigma_nu*beta_2sls[1]
    print(sigma_f)
    Var_beta_f = sigma_f * np.linalg.inv(np.dot(X1_hat.T, X1_hat))
    print(Var_beta_f)
    print(np.sqrt(np.diag(Var_beta_f)))
    
    return beta_ts2sls, Var_beta_2sls


# In sample1, we don't have mpg
# In sample 2, we don't have y
Z2 = df2[['const', 'headroom', 'weight']].to_numpy()
Z1 = df1[['const', 'headroom', 'weight']].to_numpy()
X2 = df2[['const', 'mpg', 'weight']].to_numpy()
y1 = df1[['price']].to_numpy()
y_z = df2[['mpg']].to_numpy()
beta_ts2sls, Var_beta_2sls = ts2sls(X2, Z2, Z1, y1, y_z)



def ts2sls_t3(X2, X1, Z2, Z1, y1, y_z):
    tmp_z2t2_inv = np.linalg.inv(np.dot(Z2.T, Z2))
    tmp_z2tx2 = np.dot(Z2.T, X2)
    X1_hat = np.dot(Z1, np.dot(tmp_z2t2_inv, tmp_z2tx2))
    print(X1_hat.shape)
    beta_ts2sls = np.dot(np.linalg.inv(np.dot(X1_hat.T, X1_hat)), np.dot(X1_hat.T, y1))
    print('check 0')
    print(beta_ts2sls)
    
    n1 = Z1.shape[0]
    n2 = Z2.shape[0]
    
    beta_1s = reg(Z2, y_z)
    print(beta_1s)
    pred_y_z = np.dot(Z1, beta_1s)
    print(pred_y_z.shape)
    
    print('check 1')
    print(beta_ts2sls)
    
    #pred_X1 = np.concatenate((Z1[:, 0], pred_y_z, Z1[:, 2]), axis=1)
    pred_X1 = deepcopy(X1).astype('float64')
    pred_X1[:, 2] = pred_y_z[:,0]
    #print('predicted X1')
    #print(pred_X1)
    
    print('check 2')
    print(beta_ts2sls)
    k_p = pred_X1.shape[1]
    eps = y1 - np.dot(pred_X1, beta_ts2sls)
    #eps = y1 - np.dot(X2, beta_ts2sls)
    sigma_2 = np.dot(eps.T, eps)/(n1 - k_p)
    print(sigma_2)

    Var_beta_2sls = sigma_2 * np.linalg.inv(np.dot(X1_hat.T, X1_hat))
    print(Var_beta_2sls)
    print(np.sqrt(np.diag(Var_beta_2sls)))
    
    k_q = Z2.shape[1]
    pred_X2 = deepcopy(X2).astype('float64')
    pred_y_z2 = np.dot(Z2, beta_1s)
    #print(pred_y_z2)
    pred_X2[:, 2] = pred_y_z2[:, 0]
    eps_1s = X2 - pred_X2
    print(np.sum(eps_1s, axis=0))
    sigma_nu = np.dot(eps_1s.T, eps_1s)/(n2-k_q)
    print(sigma_nu)

    print(sigma_2)
    print(beta_ts2sls)    
    sigma_f = sigma_2 + n1/n2 * np.dot(beta_ts2sls.T, np.dot(sigma_nu, beta_ts2sls))
    print(sigma_f)
    Var_beta_f = sigma_f * np.linalg.inv(np.dot(X1_hat.T, X1_hat))
    print(Var_beta_f)
    print(np.sqrt(np.diag(Var_beta_f)))
    
    return beta_ts2sls, Var_beta_2sls



df1['weight2'] = df1['weight'] * df1['weight']
df2['weight2'] = df2['weight'] * df2['weight']



Z2 = df2[['const', 'weight', 'headroom']].to_numpy()
Z1 = df1[['const', 'weight', 'headroom']].to_numpy()
X2 = df2[['const', 'weight', 'mpg']].to_numpy()
X1 = df1[['const', 'weight', 'mpg']].to_numpy()
y1 = df1[['price']].to_numpy()
y_z = df2[['mpg']].to_numpy()
beta_ts2sls, Var_beta_2sls = ts2sls_t3(X2, X1, Z2, Z1, y1, y_z)
print('We are here')





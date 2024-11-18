import os
# import numpy as np
import autograd.numpy as np
import pandas as pd
import plotly.graph_objects as go
#from PMMR.util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
#    remove_outliers, nystrom_decomp_from_orig, nystrom_decomp_from_sub, chol_inv, bundle_az_aw, visualise_ATEs, data_transform, data_inv_transform, indicator_kern
from util_R import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp_from_orig, nystrom_decomp_from_sub, chol_inv, bundle_az_aw, visualise_ATEs, data_transform, data_inv_transform, indicator_kern
# from simulation_arthur import data_transform
from datetime import date
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
import argparse
import random
import time
import autograd.numpy as anp
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy import integrate
import matplotlib.pyplot as plt

# Global parameters

Nfeval = 1
JITTER_W, JITTER_L, JITTER_LW = 1e-12, 1e-5, 1e-7



train_sizes = [300]
test_sz, dev_sz = 100, 40
nystr_M = 3000
log_al_bounds, log_bl_bounds = np.array([-1.5, 1.5]), np.array([-4., 2.])
nystr_thresh = 5000
seed = 120

# data_seeds = np.int_(np.linspace(100, 896, 200))

al_diff_search_range = [-0., -0., 1]
bl_search_range = [-4, 2., 30]
wx_dim = 3




class parameters():
    def __init__(self, sem, hparam, selection_metric, cond_metric, supp_test, log_al_bounds, log_bl_bounds, nystr_M, offset, lmo):
        self.sem = sem
        self.hparam = hparam
        self.selection_metric = selection_metric
        self.cond_metric = cond_metric
        self.supp_test = supp_test
        self.log_al_bounds = log_al_bounds
        self.log_bl_bounds = log_bl_bounds
        self.nystr_M = nystr_M
        self.offset = offset
        self.lmo = lmo
        

def compute_bandwidth_median_dist_ver2(X):
  # use the lower triangle of the matrix
    sqdist = pairwise_distances(X, Y=None, metric='euclidean')
    dist = sqdist[np.tril_indices(n=X.shape[0], k=-1)]
    al_default = np.median(dist)
    print('al default: ', al_default)
    return al_default

# Compute the bandwidth of kernel for functional treatment. The trapezoid integration rule is used
# only use the lower triangular part of the distance matrix.
def compute_bandwidth_median_dist_for_A(A):

    sqdist = []
    t = np.linspace(0, 1, num=A.shape[1])
    
    for i in range(A.shape[0]):
      for j in range(i):
        diffs = A[i,:] - A[j,:]
        #sqdist_0 = np.mean(diffs.squeeze()**2)
        sqdist_0 = integrate.trapezoid(diffs**2, t)
        sqdist.append(sqdist_0)
    dist = np.sqrt(sqdist)
    print("al default for A:", np.median(dist))
    return np.median(dist)


#def make_gaussian_prodkern(arr1, arr2, sigma):
#    dims = arr1.shape[-1]
#    assert arr1.shape[-1] == arr2.shape[-1]

#    K = 1
#    for dim in range(dims):
#        K_0 = _sqdist(arr1[:, dim].reshape(-1, 1), arr2[:, dim].reshape(-1, 1))
#        sig = sigma[dim]
#        if (type(sig) is not np.float64) and (type(sig) is not np.float32):
#            from math import e
#            print('K dim {}: '.format(dim), np.linalg.cond(e ** (-K_0 / sig._value / sig._value / 2)))
#        else:
#            print('K dim {}: '.format(dim), np.linalg.cond(np.exp(-K_0 / sig / sig / 2)))
#        K = K * anp.exp(-K_0 / sig / sig / 2)
#        del K_0
#    return K


def make_gaussian_prodkern_ver2(arr1, arr2, sigma):
    #dims = arr1.shape[-1]
    assert arr1.shape[-1] == arr2.shape[-1]
    sqdist = pairwise_distances(X=arr1, Y=arr2, metric='euclidean')
    K = np.exp(-sqdist**2/ sigma / sigma/2)
    return K    

def make_gaussian_kern_A(A1, A2, sigma_a):
    assert A1.shape[-1] == A2.shape[-1]
    t = np.linspace(0, 1, num=A1.shape[1])
    sqdist = np.zeros((A1.shape[0], A2.shape[0]))
    for i in range(A1.shape[0]):
      for j in range(A2.shape[0]):
        diffs = A1[i,:] - A2[j,:]
        sqdist[i,j] = integrate.trapezoid(diffs**2, t)
        
    K_A = np.exp(-sqdist/ sigma_a / sigma_a/ 2)
    return K_A
    

def make_prodkern_with_functional_treat(arr1, arr2, sigma_wx, sigma_a, wx_dim):
    dims = arr1.shape[-1]
    arr1_wx = arr1[:,:wx_dim]
    arr2_wx = arr2[:,:wx_dim]
    K_wx = make_gaussian_prodkern_ver2(arr1_wx, arr2_wx, sigma_wx)
    A1 = arr1[:,wx_dim:]
    A2 = arr2[:,wx_dim:]
    K_A = make_gaussian_kern_A(A1, A2, sigma_a)
    K = K_wx*K_A
    return K


def process_data_functional_treat(train_size, dev_size, test_size, args, data_seed, LOAD_PATH):
    t1 = time.time()

    # loads all data
    sim_data = pd.read_csv(os.path.join(LOAD_PATH, "sim_{}.csv".format(data_seed)), header=0).values
    train_WX, train_ZX, train_Y = sim_data[:train_size, [1,4,5]], sim_data[:train_size, 3:], sim_data[:train_size, 0].reshape(-1,1)
    test_WX, test_ZX, test_Y = sim_data[train_size:,[1,4,5]], sim_data[train_size:, 3:], sim_data[train_size:, 0].reshape(-1,1)

    # Standardize WX and ZX
    train_WX_scaled, WX_scaler = data_transform(train_WX)
    test_WX_scaled = WX_scaler.transform(test_WX)
    train_ZX_scaled, ZX_scaler = data_transform(train_ZX)
    test_ZX_scaled = ZX_scaler.transform(test_ZX)

    do_A = pd.read_csv(os.path.join(LOAD_PATH, "do_A_{}.csv".format(data_seed)), header=0).values
    do_A_train, do_A_test = do_A[:train_size, :], do_A[train_size:, :]

    train_X, train_Z = np.concatenate((train_WX_scaled, do_A_train), axis=1), np.concatenate((train_ZX_scaled, do_A_train), axis=1)
    test_X, test_Z = np.concatenate((test_WX_scaled, do_A_test), axis=1), np.concatenate((test_ZX_scaled, do_A_test), axis=1)

    EY_do_A_gt = pd.read_csv(os.path.join(LOAD_PATH, "EY_do_A_{}.csv".format(data_seed)), header=0).values
    EY_do_A_gt_train, EY_do_A_gt_test = EY_do_A_gt[:train_size, :], EY_do_A_gt[train_size:, :]

    #W_marginal = train.w[:args.w_marginal_size].reshape(args.w_marginal_size, -1)
    WX_marginal_train = train_WX_scaled
    WX_marginal_test = test_WX_scaled
    wx_dim = WX_marginal_train.shape[-1]
    zx_dim = train_ZX.shape[-1]

    t2 = time.time()
    print('data loading used {}s'.format(t2 - t1))
    return train_X, train_Y, train_Z, test_X, test_Y, test_Z, WX_marginal_train, WX_marginal_test, do_A_train, do_A_test, EY_do_A_gt_train, EY_do_A_gt_test, wx_dim, zx_dim


#def compute_alpha(train_size, eig_vec_K, W_nystr, X, Y, W, eig_val_K, nystr, params_l):
def compute_alpha(train_size, eig_vec_K, W_nystr, X, Y, W, eig_val_K, nystr, params_l, sigma_a, wx_dim):
    N2 = train_size ** 2
    EYEN = np.eye(train_size)

    al, bl = params_l
    print('al, bl = ', params_l)

    print('making K_L')
    #K_L = make_gaussian_prodkern(X, X, al)
    t1 = time.time()
    K_L = make_prodkern_with_functional_treat(X, X, al, sigma_a, wx_dim)
    t2 = time.time()
    print('end of making K_L')
    print('making K_L used {}s'.format(t2 - t1))

    L = bl * bl * K_L + JITTER_L * EYEN  # L = (1/lambda * 1/(n^2)) * L_true
    # L = bl * bl * K_L
    print('bl * bl * K_L: ', bl * bl * K_L[:10, :10])
    print('L[:10, :10]: ', L[:10, :10])

    if nystr:
        tmp = eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)
        print('condition number of 1st term: ', np.linalg.cond(eig_vec_K.T @ L @ eig_vec_K / N2, p=2))
        print('condition number of tmp: ', np.linalg.cond(tmp, p=2))
        # print('condition number of tmp not divided by N2: ', np.linalg.cond(eig_vec_K.T @ L @ eig_vec_K + np.diag(1 / eig_val_K)))
        print('tmp: ', tmp)
        print('condition number of V~: ', np.linalg.cond(np.diag(1 / eig_val_K / N2), p=2))
        print('V~: ', np.diag(1 / eig_val_K / N2))
        del tmp
        alpha = EYEN - eig_vec_K @ np.linalg.inv(
            eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
        alpha = alpha @ W_nystr @ Y * N2
    else:
        print('L condition number: ', np.linalg.cond(L))
        print('W condition number: ', np.linalg.cond(W))
        print('L @ W @ L + L / N2 condition number: ', np.linalg.cond(L @ W @ L + L / N2))
        print('L @ W @ L + L / N2 + JITTER_LW * EYEN condition number: ', np.linalg.cond(L @ W @ L + L / N2 + JITTER_LW * EYEN))
        # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
        LWL_inv = chol_inv(L @ W @ L + L / N2)
        alpha = LWL_inv @ L @ W @ Y

    return alpha


def get_causal_effect(do_A, WX_marginal, X, wx_dim, alpha, params_l, sigma_a, offset=0):
    "to be called within experiment function."
    assert WX_marginal.ndim == 2
    assert do_A.ndim == 2
    al, bl = params_l
    print('WX_shape: ', WX_marginal.shape[0])
    
    print('making ate.')

    do_A_rep = np.repeat(do_A, repeats=WX_marginal.shape[0], axis=0)
    wx_rep = np.tile(WX_marginal, [do_A.shape[0], 1])
    awx_rep = np.concatenate([wx_rep, do_A_rep], axis=-1)
    
    print('making K_L ate.')
    t1 = time.time()
    K_L_ate = make_prodkern_with_functional_treat(awx_rep, X, al, sigma_a, wx_dim=wx_dim)
    t2 = time.time()
    print('making K_L used {}s'.format(t2 - t1))
    
    ate_L = bl * bl * K_L_ate  # lambda = 1/(b_l^2n^2), b_l^2 = 1/(lambda n^2)
    print('end of making K_L ate.')
    h_out = ate_L @ alpha

    h_out_a_as_rows = h_out.reshape(-1, WX_marginal.shape[0])
    ate_est = np.mean(h_out_a_as_rows, axis=1).reshape(-1,1) + offset
    
    print('Estimated ATE: ', ate_est)
    #ate_est = np.array(ate_est)
    
    return ate_est
  
  
def get_causal_effect_no_crossfit(do_A, WX_marginal, X, Y, W_, wx_dim, params_l, sigma_a, offset=0, k=3):
  
    assert WX_marginal.ndim == 2
    assert do_A.ndim == 2
    # al, bl = params_l_final
    print('WX_shape: ', WX_marginal.shape[0])
    
    print('making ate.')
    t1 = time.time()
    ate_est = []
    
    kf = KFold(n_splits=k, random_state=None)
    
    for train_index, test_index in kf.split(do_A):
      WX_train, WX_test = WX_marginal[train_index,:], WX_marginal[test_index,:]
      X_train, X_test = X[train_index,:], X[test_index,:]
      Y_train, Y_test = Y[train_index, :], Y[test_index,:]
      W_train = W_[np.ix_(train_index, train_index)] / (X_train.shape[0] ** 2)
      

      alpha = compute_alpha(train_size = X_train.shape[0], eig_vec_K=None, W_nystr=None, X=X_train, Y=Y_train, W=W_train, eig_val_K=None, nystr=False, 
                            params_l=params_l, sigma_a=sigma_a, wx_dim=wx_dim)
      
      est_causal_effect = get_causal_effect(do_A=do_A, WX_marginal=WX_test, X=X_train, wx_dim=wx_dim, alpha=alpha, params_l=params_l, sigma_a=sigma_a, offset=offset)
      ate_est.append(est_causal_effect)
      
    
    ate_est_no_crossfit = np.array(ate_est)
    # ate_est_ave = np.mean(ate_est_crossfit, axis=0)
    
    
    print('Estimated ATE: ', ate_est_no_crossfit)
    
    return ate_est_no_crossfit



def mmr_loss(ak, al, bl, sigma_a, alpha, y_test, aw_test, az_test, X, wx_dim, zx_dim, offset):
    print('making K_L_mse not supported.')
    K_L_mse = make_prodkern_with_functional_treat(aw_test, X, al, sigma_a, wx_dim)
    print('end of making K_L_mse not supported.')
    mse_L = bl * bl * K_L_mse
    mse_h = mse_L @ alpha + offset

    print('supp_az shape: ', az_test.shape)
    N = y_test.shape[0]
    print('making K_W test.')
    K = make_prodkern_with_functional_treat(az_test, az_test, ak, sigma_a, zx_dim)
    print('end of making K_W test.')

    W_U = (K - np.diag(np.diag(K)))
    W_V = K

    assert y_test.ndim > 1
    assert mse_h.ndim == y_test.ndim
    for dim in range(mse_h.ndim):
        assert mse_h.shape[dim] == y_test.shape[dim]

    d = mse_h - y_test

    loss_V = d.T @ W_V @ d / N / N
    loss_U = d.T @ W_U @ d / N / (N - 1)
    return loss_V[0, 0], loss_U[0, 0]


def LMO_err_global(log_params_l, sigma_a, train_size, W, W_nystr_Y, eig_vec_K, inv_eig_val_K, X, Y, wx_dim, nystr, offset, M=10):
    EYEN = np.eye(train_size)
    N2 = train_size ** 2

    log_al, log_bl = log_params_l[:-1], log_params_l[-1]
    al, bl = anp.exp(log_al).squeeze(), anp.exp(log_bl).squeeze()
    # print('lmo_err params_l', params_l)
    print('lmo_err al, bl', al, bl)
    K_L = make_prodkern_with_functional_treat(X, X, al, sigma_a, wx_dim)
    L = bl * bl * K_L + JITTER_L * EYEN
    # L = bl * bl * K_L
    print('condition number of L: ', np.linalg.cond(L, p=2))
    if nystr:
        tmp_mat = L @ eig_vec_K
        C = L - tmp_mat @ anp.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
        c = C @ W_nystr_Y * N2
    else:
        # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
        LWL_inv = chol_inv(L @ W @ L + L / N2)
        C = L @ LWL_inv @ L / N2
        c = C @ W @ Y * N2
    c_y = c - Y
    lmo_err = 0
    N = 0
    for ii in range(1):
        idxs = np.arange(X.shape[0])
        for i in range(0, X.shape[0], M):
            indices = idxs[i:i + M]
            K_i = W[np.ix_(indices, indices)] * N2
            C_i = C[np.ix_(indices, indices)]
            c_y_i = c_y[indices]
            b_y = anp.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
            lmo_inc = (b_y - offset).T @ K_i @ (b_y - offset)
            lmo_err = lmo_err + lmo_inc
            # print('lmo_inc: ', lmo_inc)
            N += 1
    print('LMO-err: ', lmo_err[0, 0] / N / M ** 2)
    return lmo_err[0, 0] / N / M ** 2


def compute_losses(params_l, ax, w_samples, y_samples, y_axz, x_on,
                   AW_test, AZ_test, Y_test, supp_y, supp_aw, supp_az,
                   X, Y, Z, W, W_nystr_Y, eig_vec_K, inv_eig_val_K, nystr, test_Y, ak, sigma_a, alpha, offset, wx_dim,zx_dim, args):
    "to calculated the expected error E_{A,X,Z ~ unif}[E[Y - h(A,X,W)|A,X,Z]]."

    al, bl = params_l
    args.al = al
    args.bl = bl

    if args.cond_metric:
        if not x_on:
            ax = ax[:, 0:1]

        num_reps = w_samples.shape[1] // w_dim
        assert len(ax.shape) == 2
        assert ax.shape[1] < 3
        assert ax.shape[0] == w_samples.shape[0]
        # print('number of points: ', w_samples.shape[0])

        ax_rep = np.repeat(ax, [num_reps], axis=0)
        assert ax_rep.shape[0] == (w_samples.shape[1] * ax.shape[0])

        w_samples_flat = w_samples.flatten().reshape(-1, w_dim)
        axw = np.concatenate([w_samples_flat, ax_rep], axis=-1)

        K_L_axw = make_gaussian_prodkern(axw, X, al)
        expected_err_L = bl * bl * K_L_axw
        h_out = expected_err_L @ alpha + offset

        h_out = h_out.reshape([-1, w_samples.shape[1]//w_dim])
        y_axz_recon = np.mean(h_out, axis=1)
        assert y_axz_recon.shape[0] == y_axz.shape[0]
        mean_sq_error = np.mean(np.square(y_axz - y_axz_recon))

        # for debugging compute the mse between y samples and h
        y_samples_flat = y_samples.flatten()
        mse_alternative = np.mean((y_samples_flat - h_out.flatten()) ** 2)
    else:
        mean_sq_error, mse_alternative, y_axz_recon = None, None, None

    # standard mse
    K_L_mse = make_prodkern_with_functional_treat(AW_test, X, al, sigma_a=sigma_a, wx_dim=wx_dim)
    mse_L = bl * bl * K_L_mse
    mse_h = mse_L @ alpha + offset
    mse_standard = np.mean((test_Y.flatten() - mse_h.flatten()) ** 2)

    # standard mse on support
    if args.supp_test:
        mse_supp = compute_loss_on_supported_test_set(X=X, al=al, bl=bl, alpha=alpha,
                                                      supp_y=supp_y, supp_aw=supp_aw, supp_az=supp_az, offset=offset)
        mmr_v_supp, mmr_u_supp = mmr_loss(ak=ak, al=al, bl=bl, sigma_a=sigma_a, alpha=alpha, y_test=supp_y, aw_test=supp_aw,
                                          az_test=supp_az, X=X, wx_dim=wx_dim, zx_dim=zx_dim, offset=offset)
    else:
        mse_supp, mmr_v_supp, mmr_u_supp = None, None, None
    
    # mmr losses
    mmr_v, mmr_u = mmr_loss(ak=ak, al=al, bl=bl, sigma_a=sigma_a,alpha=alpha, y_test=Y_test, aw_test=AW_test,
                                          az_test=AZ_test, X=X, wx_dim=wx_dim, zx_dim=zx_dim, offset=offset)

    # lmo
    log_params = np.append(np.log(params_l[0]), np.log(params_l[1]))
    lmo_err = LMO_err_global(log_params_l=log_params, sigma_a=sigma_a, train_size=X.shape[0], W=W, W_nystr_Y=W_nystr_Y,
                             eig_vec_K=eig_vec_K, inv_eig_val_K=inv_eig_val_K, X=X, Y=Y, wx_dim=wx_dim, nystr=nystr, M=1, offset=offset)

    return {'err_in_expectation': mean_sq_error,
            'mse_alternative': mse_alternative,
            'y_axz_recon': y_axz_recon,
            'mse_standard': mse_standard,
            'mse_supp': mse_supp,
            'mmr_v_supp': mmr_v_supp,
            'mmr_v': mmr_v,
            'mmr_u_supp': mmr_u_supp,
            'mmr_u': mmr_u,
            'lmo': lmo_err}


def get_results(do_A, EYhat_do_A, EY_do_A_gt, train_sz, err_in_expectation, mse_alternative, mse_standard, mse_supp,
                mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, lmo, params_l, args, SAVE_PATH, LOAD_PATH, data_seed):

    causal_effect_mean_abs_err = np.mean(np.abs(EY_do_A_gt.squeeze() - EYhat_do_A.squeeze()))
    causal_effect_mse = np.mean((EY_do_A_gt.squeeze() - EYhat_do_A.squeeze())**2)

    causal_std = np.std(np.abs(EYhat_do_A.squeeze() - EY_do_A_gt.squeeze()))
    causal_rel_err = np.mean(np.abs((EYhat_do_A.squeeze() - EY_do_A_gt.squeeze())/EY_do_A_gt.squeeze()))

    return causal_effect_mean_abs_err, causal_effect_mse


def experiment(seed, data_seed, param_l_arg, train_size, dev_size, test_size, nystr, args, SAVE_PATH, LOAD_PATH):
    np.random.seed(seed)
    #random.seed(seed)

    #X, Y, Z, test_X, test_Y, test_Z, W_marginal, do_A, EY_do_A_gt, wx_dim = process_data(train_size=train_size, dev_size=dev_size,
    X, Y, Z, test_X, test_Y, test_Z, WX_marginal_train, WX_marginal_test, do_A_train, do_A_test, EY_do_A_gt_train, EY_do_A_gt_test, wx_dim, zx_dim = process_data_functional_treat(train_size=train_size, dev_size=dev_size,
                                                                                                                                                         test_size=test_size,
                                                                                                                                                         args=args, data_seed=data_seed,
                                                                                                                                                         LOAD_PATH=LOAD_PATH)

    w_samples, y_samples, y_axz, ax, axzy = load_err_in_expectation_metric(args, data_seed=data_seed) if args.cond_metric else (None, None, None, None, None)
    aw_test_supp, az_test_supp, y_test_supp = load_test_supp_metric(args, data_seed=data_seed) if args.supp_test else (None, None, None)

    #al_default = compute_bandwidth_median_dist(X=X[:, :wx_dim])
    al_default = compute_bandwidth_median_dist_ver2(X=X[:, :wx_dim])
    median_dist_a = compute_bandwidth_median_dist_for_A(A=X[:, wx_dim:])
    sigma_a = median_dist_a

    EYEN = np.eye(X.shape[0])

    print('Z shape: ', Z.shape)
    ak = compute_bandwidth_median_dist_ver2(Z[:,:zx_dim])
    
    N2 = X.shape[0] ** 2
    print('making W_.')
    W_ = make_prodkern_with_functional_treat(Z, Z, sigma_wx=ak, sigma_a=sigma_a, wx_dim=zx_dim) + JITTER_W * EYEN
    print('end of making W_.')
    print('W_ condition number: ', np.linalg.cond(W_, p=2))
    W = W_ / N2
    print('ak = ', ak)
    print('W[:10, :10]: ', W[:10, :10])

    # L0, test_L0 = _sqdist(X, None), _sqdist(test_X, X)

    if nystr:
        random_indices = np.sort(np.random.choice(range(W.shape[0]), args.nystr_M, replace=False))
        eig_val_K, eig_vec_K = nystrom_decomp_from_orig(W * N2, random_indices)
        inv_eig_val_K = np.diag(1 / eig_val_K / N2)
        W_nystr_ = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T
        W_nystr = W_nystr_ / N2
        print('(W_nystr sub - W)/W: ', (W_nystr_[:10, :10] - W_[:10, :10])/W_[:10, :10])
        print('sum (W_nystr sub - W)/W: ', np.sum(np.abs(W_nystr_ - W_)))
        W_nystr_Y = W_nystr @ Y
    else:
        eig_vec_K, eig_val_K, inv_eig_val_K = None, None, None
        W_nystr, W_nystr_Y = None, None


    al, bl = None, None
    if args.hparam == 'lmo':
        np.random.seed(seed)
        global Nfeval, prev_norm, opt_params_l, opt_test_err
        log_al0, log_bl0 = np.log(al_default), np.random.randn(1)
        params_l0 = np.append(log_al0, log_bl0)
        opt_params_l = (np.exp(log_al0), np.exp(log_bl0))
        prev_norm, opt_test_err = None, None
        print('starting param log_al0: {}, log_bl0: {}'.format(log_al0, log_bl0))
        bounds = None

        def LMO_err(params_l, M=10):
            log_al, log_bl = params_l[:-1], params_l[-1]
            al, bl = anp.exp(log_al).squeeze(), anp.exp(log_bl).squeeze()
            # print('lmo_err params_l', params_l)
            print('lmo_err al, bl', al, bl)
            K_L = make_prodkern_with_functional_treat(X, X, al, sigma_a, wx_dim)
            # L = bl * bl * K_L + JITTER * EYEN
            L = bl * bl * K_L + JITTER_L * EYEN

            if nystr:
                tmp_mat = L @ eig_vec_K
                C = L - tmp_mat @ anp.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
                c = C @ W_nystr_Y * N2
            else:
                # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER_L * EYEN)
                LWL_inv = chol_inv(L @ W @ L + L / N2)
                C = L @ LWL_inv @ L / N2
                c = C @ W @ Y * N2
            c_y = c - Y
            lmo_err = 0
            N = 0
            for ii in range(1):
                # permutation = np.random.permutation(X.shape[0])
                idxs = np.arange(X.shape[0])
                for i in range(0, X.shape[0], M):
                    indices = idxs[i:i + M]
                    K_i = W[np.ix_(indices, indices)] * N2
                    C_i = C[np.ix_(indices, indices)]
                    c_y_i = c_y[indices]
                    b_y = anp.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
                    lmo_inc = b_y.T @ K_i @ b_y
                    lmo_err = lmo_err + lmo_inc
                    # print('lmo_inc: ', lmo_inc)
                    N += 1
            print('LMO-err: ', lmo_err[0, 0] / N / M ** 2)
            # alpha = compute_alpha(train_size, eig_vec_K, W_nystr, X, Y, W, eig_val_K, nystr, anp.exp(params_l))
            # get_causal_effect(do_A, w_marginal, X, alpha, params_l, offset=0)
            return lmo_err[0, 0] / N / M ** 2

        def LMO_log_bl(log_bl):
            # params_l = np.array([log_al0, log_bl0])
            params_l = np.append(log_al0, log_bl)
            return LMO_err(params_l=params_l, M=10)

        def callback0(params_l, timer=None):
            global Nfeval, prev_norm, opt_params_l, opt_test_err
            # np.random.seed(3)
            # random.seed(3)
            if Nfeval % 1 == 0:
                log_al, log_bl = params_l[:-1], params_l[-1]
                al, bl = np.exp(log_al).squeeze(), np.exp(log_bl).squeeze()
                print('callback al, bl', al, bl)
                K_L = make_prodkern_with_functional_treat(arr1=X, arr2=X, sigma_wx=al, sigma_a=sigma_a, wx_dim=wx_dim)
                L = bl * bl * K_L + JITTER_L * EYEN
                # L = bl * bl * K_L
                if nystr:
                    alpha = EYEN - eig_vec_K @ np.linalg.inv(
                        eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                    alpha = alpha @ W_nystr @ Y * N2
                else:
                    # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                    LWL_inv = chol_inv(L @ W @ L + L / N2)
                    alpha = LWL_inv @ L @ W @ Y
                    # L_W_inv = chol_inv(W*N2+L_inv)
                K_L_test = make_prodkern_with_functional_treat(arr1=test_X, arr2=X, sigma_wx=al, sigma_a=sigma_a, wx_dim=wx_dim)
                test_L = bl * bl * K_L_test
                pred_mean = test_L @ alpha
                if timer:
                    return
                test_err = ((pred_mean - test_Y) ** 2).mean()  # ((pred_mean-test_Y)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
                norm = alpha.T @ L @ alpha

            Nfeval += 1
            prev_norm = norm[0, 0]
            opt_test_err = test_err
            opt_params_l = (al, bl)
            print('params_l,test_err, norm: ', opt_params_l, opt_test_err, norm[0, 0])

        def callback_log_bl(log_bl):
            global Nfeval, prev_norm, opt_params_l, opt_test_err
            # np.random.seed(3)
            # random.seed(3)
            if Nfeval % 1 == 0:
                al, bl = np.exp(log_al0).squeeze(), np.exp(log_bl).squeeze()
                print('callback al, bl', al, bl)
                K_L = make_prodkern_with_functional_treat(arr1=X, arr2=X, sigma_wx=al, sigma_a=sigma_a, wx_dim=wx_dim)
                L = bl * bl * K_L + JITTER_L * EYEN
                # L = bl * bl * K_L
                if nystr:
                    alpha = EYEN - eig_vec_K @ np.linalg.inv(
                        eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                    alpha = alpha @ W_nystr @ Y * N2
                else:
                    # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                    LWL_inv = chol_inv(L @ W @ L + L / N2)
                    alpha = LWL_inv @ L @ W @ Y
                    # L_W_inv = chol_inv(W*N2+L_inv)
                K_L_test = make_prodkern_with_functional_treat(arr1=test_X, arr2=X, sigma_wx=al, sigma_a=sigma_a, wx_dim=wx_dim)
                test_L = bl * bl * K_L_test
                pred_mean = test_L @ alpha
                test_err = ((pred_mean - test_Y) ** 2).mean()  # ((pred_mean-test_Y)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
                norm = alpha.T @ L @ alpha

            Nfeval += 1
            prev_norm = norm[0, 0]
            opt_test_err = test_err
            opt_params_l = bl
            print('params_l,test_err, norm: ', opt_params_l, opt_test_err, norm[0, 0])

        if args.lmo == 'albl':
            obj_grad = value_and_grad(lambda params_l: LMO_err(params_l))
            x0 = params_l0
            dim_bandwidth = x0.shape[0] - 1
            log_al_bounds = np.tile(args.log_al_bounds, [dim_bandwidth, 1])
            log_bl_bounds = args.log_bl_bounds
            bounds = np.concatenate((log_al_bounds, log_bl_bounds.reshape(1,-1)), axis=0)
            cb = callback0
        elif args.lmo == 'bl':
            obj_grad = value_and_grad(lambda log_bl: LMO_log_bl(log_bl))
            # x0 = np.array([log_bl0])
            x0 = log_bl0
            bounds = [args.log_bl_bounds]
            cb = callback_log_bl
        else:
            raise NotImplementedError

        try:
            res = minimize(obj_grad, x0=x0, bounds=bounds, method='L-BFGS-B',
                       jac=True, options={'maxiter': 5000}, callback=cb, tol=1e-3)
        except Exception as e:
            print(e)
        # print(res)
        if res.success==False:
            with open(os.path.join(SAVE_PATH, args.hparam, 'optimization_fail_dataseed{}.txt'.format(data_seed)), 'w') as f: 
                f.write('al: {}, bl: {}'.format(np.exp(log_al0), np.exp(res.x)))
        
        if opt_params_l is None:
            params_l_final = np.exp(params_l0)
        else:
            # params_l_final = opt_params_l
            params_l_final = np.exp(res.x)

        if args.lmo == 'albl':
            args.al_lmo = params_l_final[:-1]
            args.bl_lmo = params_l_final[1]
        else:
            args.al_lmo = np.exp(log_al0)
            args.bl_lmo = params_l_final
            params_l_final = np.append(np.exp(log_al0), params_l_final)


    elif args.hparam == 'cube':
        al, bl = al_default + param_l_arg[0], param_l_arg[1]
        print('bandwidth = ', al)
        params_l_final = [al, bl]

    elif args.hparam == 'fixed':
        params_l_final = param_l_arg
    else:
        raise NotImplementedError

    alpha = compute_alpha(train_size=train_size, eig_vec_K=eig_vec_K, W_nystr=W_nystr, X=X, Y=Y, W=W,
                          eig_val_K=eig_val_K, nystr=nystr, params_l=params_l_final, sigma_a=sigma_a, wx_dim=wx_dim)

    offset = compute_offset(X=X, W=W, Y=Y, alpha=alpha, params_l=params_l_final) if args.offset else 0
    print('******************* al, bl = {}, {}, offset = {}'.format(al, bl, offset))
    # EYhat_do_A_train = get_causal_effect(do_A=do_A_train, WX_marginal=WX_marginal_train, X=X, wx_dim=wx_dim, alpha=alpha, params_l=params_l_final, sigma_a=sigma_a, offset=offset)
    EYhat_do_A_test = get_causal_effect(do_A=do_A_test, WX_marginal=WX_marginal_test, X=X, wx_dim=wx_dim, alpha=alpha, params_l=params_l_final, sigma_a=sigma_a, offset=offset)

    # EYhat_do_A_train = get_causal_effect_crossfit(do_A=do_A_train, WX_marginal=WX_marginal_train, X=X, Y=Y, W_=W_, wx_dim=wx_dim, params_l=params_l_final, sigma_a=sigma_a, offset=offset, k=3)
    causal_effect_no_cf = get_causal_effect_no_crossfit(do_A=do_A_train, WX_marginal=WX_marginal_train, X=X, Y=Y, W_=W_, wx_dim=wx_dim, params_l=params_l_final, sigma_a=sigma_a, offset=offset, k=3)
    EYhat_do_A_train = causal_effect_no_cf[1,:]
    
    losses = compute_losses(params_l=params_l_final, ax=ax, w_samples=w_samples, y_samples=y_samples, y_axz=y_axz,
                                            x_on=False, AW_test=test_X, AZ_test=test_Z, Y_test=test_Y,
                                            supp_aw=aw_test_supp, supp_y=y_test_supp, supp_az=az_test_supp, X=X, Y=Y, Z=Z,
                                            W=W, W_nystr_Y=W_nystr_Y, eig_vec_K=eig_vec_K, inv_eig_val_K=inv_eig_val_K,
                                            nystr=nystr, test_Y=test_Y, ak=ak, sigma_a=sigma_a, alpha=alpha, offset=offset, wx_dim=wx_dim, zx_dim=zx_dim, args=args)

    causal_effect_mae_train, causal_effect_mse_train = get_results(do_A=do_A_train, EYhat_do_A=EYhat_do_A_train, EY_do_A_gt=EY_do_A_gt_train, train_sz=train_size,
                                             err_in_expectation=losses['err_in_expectation'], mse_alternative=losses['mse_alternative'],
                                             mse_standard=losses['mse_standard'], mse_supp=losses['mse_supp'],
                                             mmr_v_supp=losses['mmr_v_supp'], mmr_v=losses['mmr_v'],
                                             mmr_u_supp=losses['mmr_u_supp'], mmr_u=losses['mmr_u'], lmo=losses['lmo'],
                                             params_l=params_l_final, args=args,
                                             SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH, data_seed=data_seed)
    causal_effect_mae_test, causal_effect_mse_test = get_results(do_A=do_A_test, EYhat_do_A=EYhat_do_A_test, EY_do_A_gt=EY_do_A_gt_test, train_sz=train_size,
                                             err_in_expectation=losses['err_in_expectation'], mse_alternative=losses['mse_alternative'],
                                             mse_standard=losses['mse_standard'], mse_supp=losses['mse_supp'],
                                             mmr_v_supp=losses['mmr_v_supp'], mmr_v=losses['mmr_v'],
                                             mmr_u_supp=losses['mmr_u_supp'], mmr_u=losses['mmr_u'], lmo=losses['lmo'],
                                             params_l=params_l_final, args=args,
                                             SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH, data_seed=data_seed)

    return causal_effect_mae_train, causal_effect_mae_test, losses['err_in_expectation'], al_default, EYhat_do_A_train, \
           EYhat_do_A_test, losses['mse_standard'], losses['mse_supp'], losses['mmr_v_supp'], losses['mmr_v'], losses['mmr_u_supp'], losses['mmr_u'], losses['lmo']


def do_bl_hparam_analysis_plots(SAVE_PATH, LOAD_PATH, args, train_size, bl_min, bl_max, bl_mesh_size, data_seed, **h_param_results_dict): 
  
    """
    Compare which loss has the same trend as MAE when lambda changes
    """
    #deltas = np.linspace(bl_min, bl_max, bl_mesh_size)**2
    deltas = np.exp(np.linspace(bl_min, bl_max, bl_mesh_size))**2
    ldas = 1/deltas/train_size/train_size
    print('ldas: ', ldas)

    print('plotting')
    os.makedirs(os.path.join(SAVE_PATH, args.hparam, args.sem+'_seed'+str(data_seed)), exist_ok=True)
    causal_mae = np.array(h_param_results_dict['causal_mae_train'])[:, -1]
    mean_causal_mae = np.mean(causal_mae)
    causal_mae_rescaled = 1/mean_causal_mae * causal_mae
    print('keys: ', h_param_results_dict.keys())
    for var_str in h_param_results_dict.keys():
        print(var_str)
        print((var_str == 'al_default') or (var_str=='causal_mae_train') or (var_str=='ate_est_train') or (var_str=='ate_est_test'))
        boolean = (var_str == 'al_default') or (var_str=='causal_mae_train') or (var_str=='ate_est_train') or (var_str=='ate_est_test')
        if boolean:
            continue
        var = np.array(h_param_results_dict[var_str])[:, -1]
        if var[0] is None:
            continue
        mean_var = np.mean(var)
        var_rescaled = 1/np.abs(mean_var) * var
        length = var.shape[0]
        plt.figure()
        plt.plot(ldas, var_rescaled, label=var_str)
        plt.plot(ldas, causal_mae_rescaled, label='causal_mae')
        plt.xlim(max(ldas), min(ldas))
        plt.xlabel('Hyperparameter labels'), plt.ylabel('causal_MAE_TRAIN/{}-rescaled'.format(var_str)), plt.legend()
        print('save path: ', os.path.join(SAVE_PATH, args.hparam, args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}.png'.format(var_str, train_size, args.offset)))
        plt.savefig(os.path.join(SAVE_PATH, args.hparam, args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}.png'.format(var_str, train_size, args.offset)))
        plt.close()

        plt.figure()
        plt.plot(np.arange(length), var_rescaled, label=var_str)
        plt.plot(np.arange(length), causal_mae_rescaled, label='causal_mae')
        plt.xlabel('Hyperparameter labels'), plt.ylabel('causal_MAE_TRAIN/{}-rescaled'.format(var_str)), plt.legend()
        print('save path: ', os.path.join(SAVE_PATH, args.hparam, args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}_inversc.png'.format(var_str, train_size, args.offset)))
        plt.savefig(os.path.join(SAVE_PATH, args.hparam, args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}_inversc.png'.format(var_str, train_size, args.offset)))
        plt.close()



def rank_nums_in_array(arr):
    '''
    helper to rank the numbers in an array. e.g. input = np.arr([2,1,3]), output = np.arr([1,0,2]).
    '''
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))

    return ranks


def hparam_selection_from_metric_votes(mmr_v, mmr_v_supp, lmo, hparam_arr, args):
    ranks = None
    if args.selection_metric == 'mmr_v':
        ranks_mmr_v = rank_nums_in_array(mmr_v.squeeze())
        ranks = ranks_mmr_v
    elif args.selection_metric == 'mmr_v_supp':
        ranks_mmr_v_supp = rank_nums_in_array(mmr_v_supp.squeeze())
        ranks = ranks_mmr_v_supp
    elif args.lmo == 'lmo':
        ranks_lmo = rank_nums_in_array(lmo.squeeze())
        ranks = ranks_lmo
    # sum_ranks = ranks_mmr_v + ranks_mmr_v_supp + ranks_lmo
    sum_ranks = ranks
    idx_argmin = sum_ranks.argmin()
    return hparam_arr[idx_argmin]



def get_best_hparam(results_dict, args):
    hparam_arr = np.array(results_dict['causal_mae_train'])[:, :2]
    mmr_vs = np.array(results_dict['mmr_v'])[:, -1]
    mmr_v_supps = np.array(results_dict['mmr_v_supp'])[:, -1]
    lmos = np.array(results_dict['lmo'])[:, -1]
    return hparam_selection_from_metric_votes(mmr_v=mmr_vs, mmr_v_supp=mmr_v_supps, lmo=lmos, hparam_arr=hparam_arr, args=args)


def cube_search(al_diff_min, al_diff_max, al_mesh_size, bl_min, bl_max, bl_mesh_size, args, SAVE_PATH, LOAD_PATH, train_size, seed, data_seed):
    #al_arr, bl_arr = np.linspace(al_diff_min, al_diff_max, al_mesh_size), np.linspace(bl_min, bl_max, bl_mesh_size)
    al_arr, bl_arr = np.linspace(al_diff_min, al_diff_max, al_mesh_size), np.exp(np.linspace(bl_min, bl_max, bl_mesh_size))
    results_dict = {'causal_mae_train': [],
                    # 'causal_mae_test': [],
                    # 'err_in_expectation': [],
                    # 'mse_supp': [],
                    # 'mse_standard': [],
                    'mmr_v_supp': [],
                    'mmr_v': [],
                    # 'mmr_u_supp': [],
                    # 'mmr_u': [],
                    'lmo': [],
                    'ate_est_train': {},
                    'ate_est_test': {}}

    for al in al_arr:
        for bl in bl_arr:
            params_l = [al, bl]
            print('fitting al = {}, bl = {}'.format(al, bl))
            
           
            causal_effect_mae_train, causal_effect_mae_test, err_in_expectation, al_default, causal_effect_est_train, causal_effect_est_test, mse_standard, mse_supp, \
            mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, lmo = experiment(seed=seed, data_seed=data_seed, param_l_arg=params_l,
                                                                   train_size=train_size, test_size=test_sz, dev_size=dev_sz,
                                                                   nystr=(False if train_size <= nystr_thresh else True),
                                                                   args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH)

            
            if 'al_default' not in results_dict:
                results_dict['al_default'] = al_default
            
            results_dict['causal_mae_train'].append(np.append(al_default + al, np.array([bl, causal_effect_mae_train])))
            #results_dict['causal_mae_test'].append([al_default + al, bl, causal_effect_mae_test])
            #results_dict['err_in_expectation'].append([al_default + al, bl, err_in_expectation])
            #results_dict['mse_supp'].append([al_default + al, bl, mse_supp])
            #results_dict['mse_standard'].append([al_default + al, bl, mse_standard])
            results_dict['mmr_v_supp'].append(np.append(al_default + al, np.array([bl, mmr_v_supp])))
            results_dict['mmr_v'].append(np.append(al_default + al, np.array([bl, mmr_v])))
            #results_dict['mmr_u_supp'].append([al_default + al, bl, mmr_u_supp])
            #results_dict['mmr_u'].append([al_default + al, bl, mmr_u])
            results_dict['lmo'].append(np.append(al_default + al, np.array([bl, lmo])))
            results_dict['ate_est_train'][tuple(np.append(np.array(al_default + al), bl))] = causal_effect_est_train
            results_dict['ate_est_test'][tuple(np.append(np.array(al_default + al), bl))] = causal_effect_est_test
    # do_bl_hparam_analysis_plots(SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH, args=args, train_size=train_size,
    #                          bl_min=bl_min, bl_max=bl_max, bl_mesh_size=bl_mesh_size,
    #                          data_seed=data_seed, **results_dict)
    best_hparams_l = get_best_hparam(results_dict, args=args)
    best_ate_est_train = results_dict['ate_est_train'][tuple(best_hparams_l)]
    best_ate_est_test = results_dict['ate_est_test'][tuple(best_hparams_l)]

    print('best mae found at params_l: {} using hparam search method: {}'.format(best_hparams_l,
                                                                                      args.hparam))
    with open(os.path.join(SAVE_PATH, args.hparam, args.sem+'_seed'+str(data_seed), 'best_params_l_cube_trainsz{}_offset{}.txt'.format(train_size, args.offset)), 'w') as f:
        f.write('best al: {}, best bl: {:.3f}'.format(best_hparams_l[0], best_hparams_l[-1]))

    return best_hparams_l, best_ate_est_train, best_ate_est_test
  


def hyparameter_selection(al_diff_min, al_diff_max, al_mesh_size, bl_min, bl_max, bl_mesh_size, args, SAVE_PATH, LOAD_PATH, train_size, seed, data_seed):
    if args.hparam == 'cube':
        best_hparams_l, best_ate_est_train, best_ate_est_test = cube_search(al_diff_min=al_diff_min, al_diff_max=al_diff_max, al_mesh_size=al_mesh_size,
                                                   bl_min=bl_min, bl_max=bl_max, bl_mesh_size=bl_mesh_size,
                                                   args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH, train_size=train_size, data_seed=data_seed, seed=seed)
        return best_hparams_l, best_ate_est_train, best_ate_est_test

    elif args.hparam == 'lmo':
      
        causal_effect_mae_train, causal_effect_mae_test, err_in_expectation, al_default, causal_effect_est_train, causal_effect_est_test, mse_standard, \
        mse_supp, mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, lmo = experiment(seed=seed, data_seed=data_seed, param_l_arg=[],
                                                                         train_size=train_size, dev_size=dev_sz, test_size=test_sz,
                                                                         nystr=(False if train_size <= nystr_thresh else True),
                                                                         args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH)

        return [args.al_lmo, args.bl_lmo], causal_effect_est_train, causal_effect_est_test
    else:
        raise NotImplementedError


def evaluate_ate_est(ate_est, ate_gt):
    causal_mse = np.mean((ate_gt.squeeze() - ate_est.squeeze())**2)
    causal_mae = np.mean(np.abs(ate_gt.squeeze() - ate_est.squeeze()))
    causal_std = np.std(np.abs(ate_est.squeeze() - ate_gt.squeeze()))
    causal_rel_err = np.mean(np.abs((ate_est.squeeze() - ate_gt.squeeze())/ate_gt.squeeze()))

    return causal_mse, causal_mae, causal_std, causal_rel_err



def run_pmmr_rkhs_noCF(seed, al_diff_search_range, bl_search_range, train_sizes, data_seeds, sname, args):
    al_diff_min, al_diff_max, al_mesh_size = al_diff_search_range
    bl_min, bl_max, bl_mesh_size = bl_search_range
    for train_size in train_sizes:
        #causal_mae_over_seeds = []
        causal_mse_over_seeds = []
        for data_seed in data_seeds:
            SAVE_PATH = os.path.join(ROOT_PATH, "PMMR results no cf/",sname)
            LOAD_PATH = os.path.join(ROOT_PATH, "simulation data/", sname)
            
            os.makedirs(os.path.join(SAVE_PATH, args.hparam, args.sem + '_seed' + str(data_seed)), exist_ok=True)
  
            #summary_file = open(
            #    os.path.join(SAVE_PATH, str(date.today()), args.sem + '_seed' + str(data_seed),
            #                 "summary_trainsz{}_nystrom_hparam{}_offset{}.txt".format(int(train_size), args.hparam, args.offset)), "w")
            #summary_file.close()
            
            do_A = pd.read_csv(os.path.join(LOAD_PATH, "do_A_{}.csv".format(data_seed)), header=0).values
            do_A_train, do_A_test = do_A[:train_size, :], do_A[train_size:, :]
 
            EY_do_A_gt = pd.read_csv(os.path.join(LOAD_PATH, "EY_do_A_{}.csv".format(data_seed)), header=0).values
            EY_do_A_train, EY_do_A_test = EY_do_A_gt[:train_size, :], EY_do_A_gt[train_size:, :]
            #EY_do_A_gt = np.load(os.path.join(LOAD_PATH, 'do_A_{}_seed{}.npz'.format(args.sem, data_seed)))[
            #    'gt_EY_do_A']
  
            best_hparams_l, best_ate_est_train, best_ate_est_test = hyparameter_selection(seed=seed, al_diff_max=al_diff_max, al_diff_min=al_diff_min,
                                                                 al_mesh_size=al_mesh_size, bl_min=bl_min, bl_max=bl_max,
                                                                 bl_mesh_size=bl_mesh_size, args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH,
                                                                 train_size=train_size, data_seed=data_seed)


            best_causal_mse_in, best_causal_mae_in, best_causal_std_in, best_causal_rel_err_in = evaluate_ate_est(ate_est=best_ate_est_train,
                                                                                     ate_gt=EY_do_A_train)

            best_causal_mse_out, best_causal_mae_out, best_causal_std_out, best_causal_rel_err_out = evaluate_ate_est(ate_est=best_ate_est_test,
                                                                                     ate_gt=EY_do_A_test)
            np.savez(os.path.join(SAVE_PATH, args.hparam, args.sem + '_seed' + str(data_seed),
                                  'mmr_res_trainsz{}_offset{}.npz'.format(train_size, args.offset)), do_A_train=do_A_train, do_A_test=do_A_test, ate_est_train=best_ate_est_train, ate_est_test=best_ate_est_test,
                     bl=best_hparams_l[1], train_sz=train_size,
                     causal_mse_in=best_causal_mse_in, causal_mse_out=best_causal_mse_out)
            
            causal_mse_over_seeds.append([best_causal_mse_in, best_causal_mse_out])
       #print('av c-MAE: ', np.mean(causal_mae_over_seeds))
          
        # convert array into dataframe
        mse_DF = pd.DataFrame(causal_mse_over_seeds, columns=["mse_in","mse_out"])
        # save the dataframe as a csv file
        mse_DF.to_csv(os.path.join(SAVE_PATH, args.hparam,'mse_no_cf_trainsz{}_{}.csv'.format(train_size, sname)), index=False)

        #print('av c-MSE: ', np.mean(causal_mse_over_seeds, axis=0))
        
    return causal_mse_over_seeds
  



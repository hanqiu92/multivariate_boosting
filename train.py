import numpy as np
import pandas as pd
import math
import time
import xgboost as xgb
import re
from scipy.linalg import eigh
from mbst import MultiNode,MultiBooster

###############################
## followings are sub-functions that are used in the training process

def find_splits_dict(df,feat_list):
    '''
    find potential split options given a dataframe of features
    '''
    feat_class_list = []
    splits_dict = {}
    for feat in feat_list:
        ## find split locations
        ## "x < value" or "x >= value"
        x_feat = df[feat]
        x_feat_valid = x_feat.loc[pd.notnull(x_feat)]
        min_value,max_value = np.min(x_feat_valid),np.max(x_feat_valid)
        x_feat = x_feat.fillna(min_value - 1)
        min_value,max_value = np.min(x_feat),np.max(x_feat)
        split_options = np.percentile(x_feat,np.arange(0,100+1,1))
        split_options = np.unique(split_options[(split_options > min_value) & (split_options <= max_value)])
        splits_dict[feat] = split_options

        ## convert feature value into bin index
        feat_class = feat+'_class'
        feat_class_list.append(feat_class)
        bins = np.concatenate([[min_value-1],split_options,[max_value+1]])
        df[feat_class] = pd.cut(df[feat],bins,right=False,labels=False)
        null_idxs = pd.isnull(df[feat_class])
        if np.sum(null_idxs) > 0:
            df.loc[null_idxs,feat_class] = np.max(df.loc[~null_idxs,feat_class])
        df[feat_class] = df[feat_class].astype(int)
        assert np.max(df[feat_class]) <= len(split_options)

    return df,feat_class_list,splits_dict

def parse_xgb(xgb_bst,dim,trees_idx=(-1,)):
    '''
    turn a xgb tree into a mbst tree (a list of MultiNode objects)
    because of the limitation of the xgb api, this function runs in O(n), where n is the current number of trees; it could run in O(1) with proper modifications of the xgb code
    '''
    trees_str = xgb_bst.get_dump('', with_stats=True)
    trees = []
    for tree_idx in trees_idx:
        tree_str = trees_str[tree_idx].split('\n')
        tree = [MultiNode(idx=idx,level=0,dim=dim) for idx in range(len(tree_str))]
        for node_str in tree_str:
            if node_str == '':
                continue
            parse = node_str.split(':')
            node_idx = int(parse[0].strip('\t'))
            stats = re.split('=|,',parse[1])
            if '[' not in node_str:
                ## leaf node
                tree[node_idx].pred_value = float(stats[-3]) * np.ones((dim,))
            else:
                ## split node
                split_info = node_str.split('[')[1].split(']')[0].split('<')
                tree[node_idx].split_feat = split_info[0]
                tree[node_idx].split_value = float(split_info[1])

                node_idx_left,node_idx_right = int(stats[1]),int(stats[3])
                tree[node_idx].childs = [tree[node_idx_left],tree[node_idx_right]]
                tree[node_idx_left].father = tree[node_idx]
                tree[node_idx_left].level = tree[node_idx].level + 1
                tree[node_idx_right].father = tree[node_idx]
                tree[node_idx_right].level = tree[node_idx].level + 1

        trees.append(tree)
    return trees

def find_max_eigen_vec(mat,n=None):
    '''
    find the max eigen vector of a squared matrix
    '''
    if n is None:
        n = mat.shape[-1]
    assert mat.shape[0] == mat.shape[1]
    value,vec = eigh(mat,subset_by_index=[n-1,n-1])
    vec = vec[:,0]
    return value,vec

def find_max_var_direc(x):
    '''
    find the max variance direction of the sample matrix $x$ (one-dimensional PCA)
    '''
    xx = np.expand_dims(x,axis=1) * np.expand_dims(x,axis=2)
    xx_sum = np.sum(xx,axis=0)
    _,direc = find_max_eigen_vec(xx_sum)
    return direc

def shift_grad(grad,hess,theta0):
    '''
    update the gradients and the hessians when the coordinates of the $\theta$ variable is shifted by $\theta_0$
    '''
    grad_shift = grad + np.dot(hess,theta0)
    return grad_shift,hess

def project(grad,hess,direc):
    '''
    project the multivariate gradients and hessians to univariate ones according to the direction direc
    '''
    grad_scalar = np.dot(grad,direc)
    hess_scalar = np.dot(np.dot(hess,direc),direc)
    return grad_scalar,hess_scalar

###############################
## followings are functions for training mbst and projected mbst

def train_multi_bst(params,df,labels,feat_list,obj_func,num_boost_round=10,verbose=True):
    dim = labels[0].shape[-1]
    mbst = MultiBooster(dim=dim,
                        lam_reg=params.get('lam_reg',1),
                        learning_rate=params.get('learning_rate',0.1),
                        max_depth=params.get('max_depth',4),
                        min_split_loss=params.get('min_split_loss',0),
                        subsample=params.get('subsample',0.7),
                        min_childs=params.get('min_childs',10))

    ## 1. generate split options and deal with unknown values
    df,feat_class_list,splits_dict = find_splits_dict(df,feat_list)
    feat_dict = dict([(feat,idx) for idx,feat in enumerate(feat_list)])
    feat_mat = df[feat_list].values
    feat_class_mat = df[feat_class_list].values

    ## 2. do boost rounds
    theta_pred = np.zeros((len(df),dim))
    tt = time.time()
    for iter_ in range(num_boost_round):
        loss,grad,hess = obj_func(theta_pred,labels)
        mbst.boost(grad,hess,feat_mat,feat_class_mat,feat_dict,feat_list,splits_dict)
        theta_pred += mbst.predict(feat_mat,feat_dict,tree_idxs=[-1])

        if verbose:
            print('iter:{}, loss={:.4f}. elapsed time:{:.2f}.'.format(iter_,loss,time.time() - tt))

    return mbst,splits_dict

def train_project_bst(params,df,labels,feat_list,obj_func,find_direc_func,honest=True,centered=True,corrected=False,num_boost_round=10,verbose=True):
    dim = labels[0].shape[-1]
    mbst = MultiBooster(dim=dim,
                        lam_reg=params.get('lam_reg',1),
                        learning_rate=params.get('learning_rate',0.1),
                        max_depth=params.get('max_depth',4),
                        min_split_loss=params.get('min_split_loss',0),
                        subsample=params.get('subsample',0.7),
                        min_childs=params.get('min_childs',10))

    ## 1. generate split options and deal with unknown values
    df,feat_class_list,splits_dict = find_splits_dict(df,feat_list)
    feat_dict = dict([(feat,idx) for idx,feat in enumerate(feat_list)])
    feat_mat = df[feat_list].values
    feat_class_mat = df[feat_class_list].values

    ## 2. do boost rounds

    ### xgb as the univariate split finder
    xgb_dtrain = xgb.DMatrix(df[feat_list])
    xgb_params = {'reg_lambda':params.get('lam_reg',1),
                  'learning_rate':params.get('learning_rate',0.1),
                  'max_depth':params.get('max_depth',4),
                  'subsample':params.get('subsample',0.7),}
    xgb_bst = xgb.Booster(xgb_params,[xgb_dtrain])

    ### other inputs
    N_sample = len(feat_mat)
    idxs_full = np.arange(N_sample,dtype=int)
    theta_pred = np.zeros((len(df),dim))
    delta_theta = np.zeros((len(df),dim))
    grad_func = lambda theta: obj_func(theta,labels)[1:]
    tt = time.time()
    for iter_ in range(num_boost_round):
        ## inner 1. use xgb to find splits and construct tree
        grad_split,hess_split = grad_func(theta_pred)
        if centered:
            g_split_sum,H_split_sum = np.sum(grad_split,axis=0),np.sum(hess_split,axis=0)
            delta_theta_mean,_ = mbst.find_theta(g_split_sum,H_split_sum)
            grad_split,hess_split = shift_grad(grad_split,hess_split,delta_theta_mean)
        else:
            delta_theta_mean = np.zeros((dim,))
        direc = find_direc_func(grad_split,hess_split,grad_func,mbst,
                                {'curr_theta':theta_pred,'delta_theta_mean':delta_theta_mean,'grad_prev':delta_theta/mbst.learning_rate})
        grad_split_scalar,hess_split_scalar = project(grad_split,hess_split,direc)

        xgb_bst.boost(xgb_dtrain, grad_split_scalar, hess_split_scalar)
        tree = parse_xgb(xgb_bst,trees_idx=[-1],dim=dim)[0]

        ## inner 2. update the tree predictions
        loss,grad,hess = obj_func(theta_pred,labels)
        if honest:
            ## update predictions with multivariate gradients and hessians
            tree[0].update_pred_values(grad,hess,mbst,feat_mat,feat_dict,idxs_full)

            ## corrector
            if corrected:
                ## decompose: delta_theta = delta_theta_mean + k * direc
                delta_theta_tmp = np.stack([node.pred_value for node in tree],axis=0)
                delta_theta_mean_tmp = np.mean(delta_theta_tmp,axis=0)
                delta_theta_tmp = delta_theta_tmp - np.expand_dims(delta_theta_mean_tmp,axis=0)
                grad_split,hess_split = shift_grad(grad,hess,delta_theta_mean_tmp)
                direc = find_max_var_direc(delta_theta_tmp)
                grad_split_scalar,hess_split_scalar = project(grad_split,hess_split,direc)

                ## use xgb to find splits and construct tree
                xgb_bst.boost(xgb_dtrain, grad_split_scalar, hess_split_scalar)
                tree = parse_xgb(xgb_bst,trees_idx=[-1],dim=dim)[0]

                ## update the tree predictions
                tree[0].update_pred_values(grad,hess,mbst,feat_mat,feat_dict,idxs_full)
        else:
            ## update predictions with projected univariate gradients and hessians
            for node in tree:
                node.pred_value = node.pred_value[0] * direc + delta_theta_mean

        mbst.trees.append(tree)
        delta_theta = mbst.predict(feat_mat,feat_dict,tree_idxs=[-1])
        theta_pred += delta_theta

        if verbose:
            print('iter:{}, loss={:.4f}. elapsed time:{:.2f}.'.format(iter_,loss,time.time() - tt))

    return mbst,splits_dict
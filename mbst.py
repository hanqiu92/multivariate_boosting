import numpy as np
import time
from numba import njit

@njit
def groupby_sum(grad_tmp,hess_tmp,class_tmp,num_class,dim):
    ## do a fast groupby operation of gradients and hessians,
    ## where the groups/classes are given by the class_tmp vector
    cnt_class,g_sum_class,H_sum_class = \
        np.zeros((num_class,)),np.zeros((num_class,dim)),np.zeros((num_class,dim,dim))
    for idx in range(len(class_tmp)):
        c_idx = class_tmp[idx]
        if 0 <= c_idx <= num_class - 1:
            cnt_class[c_idx] += 1
            for i in range(dim):
                g_sum_class[c_idx,i] += grad_tmp[idx,i]
                for j in range(dim):
                    H_sum_class[c_idx,i,j] += hess_tmp[idx,i,j]
    return cnt_class,g_sum_class,H_sum_class

def find_theta_base(g_sum,H_sum,lam_reg,dim):
    ## find -H^{-1}g with regularization lambda
    H_reg = H_sum + lam_reg * np.eye(dim)
    theta = -np.linalg.solve(H_reg,g_sum)
    loss = 0.5 * np.dot(g_sum,theta)
    return theta,loss

class MultiNode:
    '''
    Node objects in the mbst.
    '''

    def __init__(self,idx,level,dim,split_feat='',split_value=0,father=None,childs=[],pred_value=None):
        self.idx = idx
        self.level = level
        self.dim = dim
        self.split_feat = split_feat
        self.split_value = split_value
        self.father = father
        self.childs = childs
        if pred_value is None:
            self.pred_value = np.zeros((self.dim,))
        else:
            self.pred_value = pred_value

    def __repr__(self):
        return 'MultiNode(idx={},split_feat={},split_value={},pred_value={})'.format(self.idx,self.split_feat,self.split_value,self.pred_value)

    def update_pred_values(self,grad,hess,mbst,feat_mat,feat_dict,idxs):
        '''
        update (leaf) node predictions recursively, according to new gradients and hessians
        mainly used by the projected mbst method
        '''
        g_sum,H_sum = np.sum(grad[idxs],axis=0),np.sum(hess[idxs],axis=0)
        theta,_ = mbst.find_theta(g_sum,H_sum)
        self.pred_value = theta * mbst.learning_rate
        if len(self.childs) > 0:
            assert len(self.childs) == 2
            split_feat,split_value = self.split_feat,self.split_value
            feat_idx = feat_dict.get(split_feat,-1)
            if feat_mat.shape[-1] > feat_idx >= 0:
                bool_left = feat_mat[:,feat_idx][idxs] < split_value
                idxs_left,idxs_right = idxs[bool_left],idxs[~bool_left]
                self.childs[0].update_pred_values(grad,hess,mbst,feat_mat,feat_dict,idxs_left)
                self.childs[1].update_pred_values(grad,hess,mbst,feat_mat,feat_dict,idxs_right)

    def predict(self,feat_mat,feat_dict,idxs):
        '''
        make predictions recursively, given new features
        '''
        pred_results = []
        if len(self.childs) > 0:
            assert len(self.childs) == 2
            split_feat,split_value = self.split_feat,self.split_value
            feat_idx = feat_dict.get(split_feat,-1)
            if feat_mat.shape[-1] > feat_idx >= 0:
                bool_left = feat_mat[:,feat_idx][idxs] < split_value
                idxs_left,idxs_right = idxs[bool_left],idxs[~bool_left]
                pred_results.extend(self.childs[0].predict(feat_mat,feat_dict,idxs_left))
                pred_results.extend(self.childs[1].predict(feat_mat,feat_dict,idxs_right))
            else:
                pred_results = [(idxs,self.pred_value)]
        else:
            pred_results = [(idxs,self.pred_value)]
        return pred_results

class MultiBooster:
    def __init__(self,dim=1,lam_reg=1,learning_rate=0.1,max_depth=4,min_split_loss=0,subsample=0.5,min_childs=10,debug_display=False):
        self.trees = []
        self.dim = dim
        self.lam_reg = lam_reg
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_split_loss = min_split_loss
        self.subsample = subsample
        self.min_childs = min_childs
        self.debug_display = debug_display
        self.find_theta = lambda g_sum,H_sum: find_theta_base(g_sum,H_sum,self.lam_reg,self.dim)

    def boost(self,grad,hess,feat_mat,feat_class_mat,feat_dict,feat_list,splits_dict):
        '''
        do one boost round.
        '''
        assert grad.shape[-1] == self.dim

        N_sample = len(feat_mat)
        idxs = np.arange(N_sample,dtype=int)
        if self.subsample < 1 and self.subsample > 0:
            ## sample subsamples
            idxs = np.random.choice(idxs,size=int(self.subsample * len(idxs)),replace=False)
        
        ## initialize the root node
        g_sum,H_sum = np.sum(grad[idxs],axis=0),np.sum(hess[idxs],axis=0)
        theta,loss = self.find_theta(g_sum,H_sum)
        root_node = MultiNode(idx=0,level=0,dim=self.dim,pred_value=theta*self.learning_rate)
        tree = [root_node]
        leaves = [(root_node,idxs,theta,loss)]
        curr_node_idx = 1

        while len(leaves) > 0:
            ## iterate over any remaining unprocessed nodes
            node,idxs,theta,loss = leaves.pop(0)
            grad_tmp,hess_tmp = grad[idxs],hess[idxs]
            g_sum,H_sum = np.sum(grad[idxs],axis=0),np.sum(hess[idxs],axis=0)
            
            ## search for the best split
            best_split,best_split_gain = None,-1e8
            if node.level < self.max_depth and len(idxs) > self.min_childs:
                split_counts = 0
                for feat in feat_list:
                    values = splits_dict.get(feat,[])
                    if len(values) > 0:
                        class_tmp = feat_class_mat[:,feat_dict[feat]][idxs]
                        cnt_class,g_sum_class,H_sum_class = \
                            groupby_sum(grad_tmp,hess_tmp,class_tmp,len(values)+1,self.dim)
                        g_cumsum_class,H_cumsum_class = \
                            np.cumsum(g_sum_class,axis=0),np.cumsum(H_sum_class,axis=0)

                        for class_idx,value in enumerate(values):
                            if cnt_class[class_idx] > 0: ## otherwise, can skip this class
                                split_counts += 1
                                g_sum_left,H_sum_left = g_cumsum_class[class_idx],H_cumsum_class[class_idx]
                                g_sum_right,H_sum_right = g_sum - g_sum_left,H_sum - H_sum_left
                                theta_left,loss_left = self.find_theta(g_sum_left,H_sum_left)
                                theta_right,loss_right = self.find_theta(g_sum_right,H_sum_right)
                                split_gain = loss - (loss_left + loss_right)

                                if self.debug_display:
                                    print(feat,value,split_gain)
                                    print(theta,theta_left,theta_right)
                                    print(loss,loss_left,loss_right,loss_left+loss_right)

                                if split_gain > best_split_gain and split_gain > self.min_split_loss:
                                    best_split_gain = split_gain
                                    best_split = (feat,value,theta_left,theta_right,loss_left,loss_right)
                                    
                                    if self.debug_display:
                                        print('better',feat,value,split_gain)

            ## execute the best split and generate child nodes
            if best_split is not None:
                split_feat,split_value,theta_left,theta_right,loss_left,loss_right = best_split
                feat_idx = feat_dict.get(split_feat,-1)
                if feat_mat.shape[-1] > feat_idx >= 0:
                    node.split_feat,node.split_value = split_feat,split_value
                    bool_left = feat_mat[:,feat_idx][idxs] < split_value
                    idxs_left,idxs_right = idxs[bool_left],idxs[~bool_left]
                    node_left = MultiNode(idx=curr_node_idx,level=node.level+1,dim=self.dim,father=node,pred_value=theta_left*self.learning_rate)
                    node_right = MultiNode(idx=curr_node_idx+1,level=node.level+1,dim=self.dim,father=node,pred_value=theta_right*self.learning_rate)
                    curr_node_idx += 2
                    node.childs = [node_left,node_right]
                    tree.extend([node_left,node_right])
                    leaves.extend([(node_left,idxs_left,theta_left,loss_left),
                                (node_right,idxs_right,theta_right,loss_right)])

        self.trees.append(tree)

    def predict(self,feat_mat,feat_dict,tree_idxs=None):
        '''
        make predictions given new features
        '''
        N_sample = len(feat_mat)
        idxs = np.arange(N_sample,dtype=int)
        pred = np.zeros((N_sample,self.dim))
        if tree_idxs is None:
            trees = self.trees
        else:
            trees = [self.trees[idx] for idx in tree_idxs]
        ## iterate over trees
        for tree in trees:
            root_node = tree[0]
            ## call the predict func of the root node to obtain predictions from this tree
            pred_results_tmp = root_node.predict(feat_mat,feat_dict,idxs)
            pred_tmp = np.zeros((N_sample,self.dim))
            for idxs_tmp,value_tmp in pred_results_tmp:
                pred_tmp[idxs_tmp] = value_tmp
            pred += pred_tmp
        return pred

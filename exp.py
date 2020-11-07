import numpy as np
from mbst import MultiNode,MultiBooster

## helper functions for setting up experiments

def generate_random_theta(theta_dim,theta_factors):
    '''
    generate a random theta according to the standard Normal distribution, and multiply it by a factor
    '''
    theta = np.random.normal(size=(theta_dim,))
    theta = theta * theta_factors
    return theta

def generate_random_split(feat_list,splits_dict):
    '''
    given the list of features and the corresponding split options, generate a random split
    '''
    split_feat = np.random.choice(feat_list)
    split_value = np.random.choice(splits_dict[split_feat])
    return split_feat,split_value

def generate_random_theta_mbst(N_trees,feat_list,splits_dict,theta_dim,theta_factors):
    '''
    generate a random multivariate booster
    '''
    trees = []
    for _ in range(N_trees):
        ## iterate over each tree

        ## first, set up the root node
        split_feat,split_value = generate_random_split(feat_list,splits_dict)
        root_node = MultiNode(idx=0,level=0,dim=theta_dim,
                              pred_value=generate_random_theta(theta_dim,theta_factors) / N_trees,
                             split_feat=split_feat,split_value=split_value)
        tree = [root_node]
        leaves = [root_node]
        curr_node_idx = 1

        while len(leaves) > 0:
            ## then, for each remaining node, generate two children if not exceed the max depth
            node = leaves.pop(0)
            if node.level < 3:
                split_feat,split_value = generate_random_split(feat_list,splits_dict)
                node_left = MultiNode(idx=curr_node_idx,level=node.level+1,dim=theta_dim,father=node,
                                      pred_value=generate_random_theta(theta_dim,theta_factors) / N_trees,
                                      split_feat=split_feat,split_value=split_value)
                split_feat,split_value = generate_random_split(feat_list,splits_dict)
                node_right = MultiNode(idx=curr_node_idx+1,level=node.level+1,dim=theta_dim,father=node,
                                       pred_value=generate_random_theta(theta_dim,theta_factors) / N_trees,
                                       split_feat=split_feat,split_value=split_value)
                curr_node_idx += 2
                node.childs = [node_left,node_right]
                tree.extend([node_left,node_right])
                leaves.extend([node_left,node_right])
        trees.append(tree)
    
    ## initialize the mbst
    mbst = MultiBooster(dim=theta_dim)
    mbst.trees = trees
    return mbst
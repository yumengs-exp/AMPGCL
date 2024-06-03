import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from torch_sparse import SparseTensor
def knn_graph(X, k=20, metric='minkowski'):
    X = X.cpu().detach().numpy()
    A = kneighbors_graph(X, n_neighbors=k, metric=metric)
    edge_index = sparse_mx_to_edge_index(A)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index

def sparse_mx_to_edge_index(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    row = torch.from_numpy(sparse_mx.row.astype(np.int64))
    col = torch.from_numpy(sparse_mx.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
 
    return edge_index

def edge_index_to_adj(edge_index, num_nodes):
    # 将 edge_index 转换为稀疏张量
    values = torch.ones(edge_index.size(1)).to(edge_index.device)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=values, sparse_sizes=(num_nodes, num_nodes)).to(edge_index.device)

    # 将稀疏张量转换为密集张量（邻接矩阵）
    adj_dense = adj.to_dense()

    return adj_dense

def rand_train_test_idx(label, train_prop=.1, valid_prop=0.6, ignore_negative=True, balance=True):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return train_idx,valid_idx, test_idx

if __name__ == '__main__':
    path = osp.join(osp.expanduser('~'), 'data', 'cora')
    dataset = Planetoid(path, 'cora')
    data = dataset[0]
    knn_graph = knn_graph(data.x)
    print(knn_graph.size())
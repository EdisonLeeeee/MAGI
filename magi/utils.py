import random
import torch
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
try:
    from sklearnex import patch_sklearn
except:
    def patch_sklearn(): return

from magi.clustering_metric import clustering_metrics
from magi.batch_kmeans_cuda import kmeans


def get_sim(batch, adj, wt=20, wl=3):
    rowptr, col, _ = adj.csr()
    batch_size = batch.shape[0]
    batch_repeat = batch.repeat(wt)
    rw = adj.random_walk(batch_repeat, wl)[:, 1:]

    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    rw = rw.t().reshape(-1, batch_size).t()

    row, col, val = [], [], []
    for i in range(batch.shape[0]):
        rw_nodes, rw_times = torch.unique(rw[i], return_counts=True)
        row += [batch[i].item()] * rw_nodes.shape[0]
        col += rw_nodes.tolist()
        val += rw_times.tolist()

    unique_nodes = list(set(row + col))
    subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

    row = [subg2g[x] for x in row]
    col = [subg2g[x] for x in col]
    idx = torch.tensor([subg2g[x] for x in batch.tolist()])

    adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                        sparse_sizes=(len(unique_nodes), len(unique_nodes)))

    adj_batch, _ = adj_.saint_subgraph(idx)
    adj_batch = adj_batch.set_diag(0.)
    # src, dst = dict_r[idx[adj_batch.storage.row()[3].item()].item()], dict_r[idx[adj_batch.storage.col()[3].item()].item()]
    return batch, adj_batch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clustering(feature, n_clusters, true_labels, kmeans_device='cpu', batch_size=100000, tol=1e-4, device=torch.device('cuda:0'), spectral_clustering=False):
    if spectral_clustering:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        print("spectral clustering on cpu...")
        patch_sklearn()
        Cluster = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=0)
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
    else:
        if kmeans_device == 'cuda':
            if isinstance(feature, np.ndarray):
                feature = torch.tensor(feature)
            print("kmeans on gpu...")
            predict_labels, _ = kmeans(
                X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device)
            predict_labels = predict_labels.numpy()
        else:
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            print("kmeans on cpu...")
            patch_sklearn()
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)

    cm = clustering_metrics(true_labels, predict_labels)
    acc, nmi, adjscore, fms, f1_macro, f1_micro = cm.evaluationClusterModelFromLabel(
        tqdm)
    return acc, nmi, adjscore, f1_macro, f1_micro


def get_mask(adj):
    batch_mean = adj.mean(dim=1)
    mean = batch_mean[torch.LongTensor(adj.storage.row())]
    mask = (adj.storage.value() - mean) > - 1e-10
    row, col, val = adj.storage.row()[mask], adj.storage.col()[
        mask], adj.storage.value()[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_


def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled

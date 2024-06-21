import csv
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_undirected, add_remaining_self_loops

from magi.utils import setup_seed, get_sim, get_mask, scale, clustering
from magi.model import Model, Encoder

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=bool, default=True, help='')
parser.add_argument('--runs', type=int, default=1, help='runs')

# dataset para
parser.add_argument('--dataset', type=str, default='Cora')

# model para
parser.add_argument('--hidden', type=str, default='512', help='GNN encoder')
parser.add_argument('--projection', type=str, default='', help='Projection')

# sample para
parser.add_argument('--wt', type=int, default=100,
                    help='number of random walks')
parser.add_argument('--wl', type=int, default=2, help='depth of random walks')
parser.add_argument('--tau', type=float, default=0.3, help='temperature')

# learning para
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--ns', type=float, default=0.5, help='')
args = parser.parse_args()


def train():
    randint = random.randint(1, 1000000)
    setup_seed(randint)
    if args.verbose:
        print('random seed : ', randint, '\n', args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = 'data/'
    if args.dataset in ['Cora', 'Citeseer']:
        dataset = Planetoid(path, args.dataset)
    elif args.dataset in ['Photo', 'Computers']:
        dataset = Amazon(path, args.dataset)
    else:
        raise RuntimeError(f"Unknown dataset {args.dataset}")

    data = dataset[0]
    x, edge_index, y = data.x, data.edge_index, data.y
    N, E = data.num_nodes, data.num_edges
    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1], sparse_sizes=(N, N))
    adj.fill_value_(1.)
    batch = torch.LongTensor(list(range(N)))
    batch, adj_batch = get_sim(batch, adj, wt=args.wt, wl=args.wl)

    mask = get_mask(adj_batch)

    hidden = list(map(int, args.hidden.split(',')))
    if args.projection == '':
        projection = None
    else:
        projection = list(map(int, args.projection.split(',')))

    encoder = Encoder(data.num_features, hidden, base_model=GCNConv,
                      dropout=args.dropout, ns=args.ns).to(device)
    model = Model(
        encoder, in_channels=hidden[-1], project_hidden=projection, tau=args.tau).to(device)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    dataset2n_clusters = {'Cora': 7,
                          'Citeseer': 6, 'Photo': 8, 'Computers': 10}
    n_clusters = dataset2n_clusters[args.dataset]

    # train
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        out = scale(out)
        out = F.normalize(out, p=2, dim=1)
        loss = model.loss(out, mask)
        loss.backward()
        optimizer.step()
        if args.verbose:
            print(f'(T) | Epoch={epoch:03d}, loss={float(loss):.4f}')

    # eval
    with torch.no_grad():
        model.eval()
        out = model(x, edge_index)
        out = scale(out)
        out = F.normalize(out, p=2, dim=1).detach().cpu()
        acc, nmi, ari, f1_macro, f1_micro = clustering(
            out.numpy(), n_clusters, y.numpy(), spectral_clustering=True)

    print(
        f'train over | ACC={acc:.4f}, NMI={nmi:.4f},  ARI={ari:.4f}, f1_macro={f1_macro:.4f}, f1_micro={f1_micro:.4f}')

    return acc, nmi, ari, f1_macro, f1_micro


def run(runs=1, result=None):
    if result:
        with open(result, 'w', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(
                ['runs', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro'])

    ACC, NMI, ARI, F1_MA, F1_MI = [], [], [], [], []
    for i in range(runs):
        print(f'\n----------------------runs {i+1: d} start')
        acc, nmi, ari, f1_macro, f1_micro = train()
        print(f'\n----------------------runs {i+1: d} over')

        if result:
            with open(result, 'a', encoding='utf-8', newline='') as f_w:
                writer = csv.writer(f_w)
                writer.writerow([i+1, acc, nmi, ari, f1_macro, f1_micro])
        ACC.append(acc)
        NMI.append(nmi)
        ARI.append(ari)
        F1_MA.append(f1_macro)
        F1_MI.append(f1_micro)
    ACC = np.array(ACC)
    NMI = np.array(NMI)
    ARI = np.array(ARI)
    F1_MA = np.array(F1_MA)
    F1_MI = np.array(F1_MI)

    print(f'mean | ACC={ACC.mean():.4f}, NMI={NMI.mean():.4f},  ARI={ARI.mean():.4f}, '
          f'f1_macro={F1_MA.mean():.4f}, f1_micro={F1_MI.mean():.4f}')
    print(f'std | ACC={ACC.std():.4f}, NMI={NMI.std():.4f},  ARI={ARI.std():.4f}, '
          f'f1_macro={F1_MA.std():.4f}, f1_micro={F1_MI.std():.4f}')

    if result:
        with open(result, 'a', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(['mean', ACC.mean(), NMI.mean(),
                            ARI.mean(), F1_MA.mean(), F1_MI.mean()])
            writer.writerow(['std', ACC.std(), NMI.std(),
                            ARI.std(), F1_MA.std(), F1_MI.std()])


if __name__ == '__main__':
    result = None
    run(args.runs, result)

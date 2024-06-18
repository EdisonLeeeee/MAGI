# MAGI
The official PyTorch implementation of "Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective (MAGI)"

# Abstract
Graph clustering, a fundamental and challenging task in graph mining, aims to classify nodes in a graph into several disjoint clusters. In recent years, graph contrastive learning (GCL) has emerged as a dominant line of research in graph clustering and advances the new state-of-the-art. However, GCL-based methods heavily rely on graph augmentations and contrastive schemes, which may potentially introduce challenges such as semantic drift and scalability issues. Another promising line of research involves adopting modularity, a popular and effective measure for community detection, as the maximization goal to guide the clustering tasks. Despite the recent progress, the underlying mechanism of neural modularity maximization is still not well understood. In this work, we dig into the hidden success of neural modularity maximization for graph clustering. Our analysis reveals the strong connections between modularity maximization and contrastive learning, where positive and negative examples are naturally defined by modularity. In light of our results, we propose a community-aware graph clustering framework, coined Magi. Magi leverages modularity maximization as contrastive pretext task to effectively uncover the underlying information of communities in graphs, while mitigating the problem of semantic drift. Extensive experiments on multiple graph datasets verify the effectiveness of Magi in terms of scalability and clustering performance compared to state-of-the-art graph clustering methods. Notably, Magi easily scales a sufficiently large graph with 100M nodes while outperforming strong baselines.

# Requirements
* ogb==1.3.6
* numpy==1.23.5
* torch==1.10.1+cu111
* torch-cluster==1.6.0
* torch_geometric==2.0.4 
* torch-scatter==2.0.9
* torch-sparse==0.6.12
* scipy==1.10.1
* scikit-learn==1.3.2 
* scikit-learn-intelex==2024.1.0

# Installation
pip install -r requirements.txt

# Graph clustering experiments

* Cora
  ```
  sss="$(date +%Y-%m-%d-%H-%s)/"
  mkdir ./log
  mkdir ./log/cora
  mkdir ./log/cora/${sss}
  python ./train_gcn.py --verbose True --log True --log_file './log/cora/'${sss} --times 10 --dataset 'Cora' --hidden '512' --wt 100 --wl 2 --tau 0.3 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3
  ```
* CiteSeer
  ```
  sss="$(date +%Y-%m-%d-%H-%s)/"
  mkdir ./log
  mkdir ./log/citeseer
  mkdir ./log/citeseer/${sss}
  python ./train_gcn.py --verbose True --log False --log_file './log/citeseer/'${sss} --times 10 --dataset 'Citeseer' --hidden '1024,512' --wt 100 --wl 3 --tau 0.9 --ns 0.5 --lr 0.0001 --epochs 400 --wd 5e-4
  ```
* Amazon-photo
  ```
  sss="$(date +%Y-%m-%d-%H-%s)/"
  mkdir ./log
  mkdir ./log/photo
  mkdir ./log/photo/${sss}
  python ./train_gcn.py --verbose True --log True --log_file './log/photo/'${sss} --times 10 --dataset 'Photo' --hidden '512' --wt 100 --wl 3 --tau 0.5 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3
  ```
* Amazon-computers
  ```
  sss="$(date +%Y-%m-%d-%H-%s)/"
  mkdir ./log
  mkdir ./log/computers
  mkdir ./log/computers/${sss}
  python ./train_gcn.py --verbose True --log True --log_file './log/computers/'${sss} --times 10 --dataset 'Computers' --hidden '1024,512' --wt 100 --wl 3 --tau 0.9 --ns 0.1 --lr 0.0005 --epochs 400 --wd 1e-3
  ```
* ogbn-arxiv
  ```
  sss="$(date +%Y-%m-%d-%H-%s)/"
  mkdir ./log
  mkdir ./log/ogbn_arxiv
  mkdir ./log/ogbn_arxiv/${sss}
  python ./train_sage.py --verbose True --log True --log_file './log/ogbn_arxiv/'${sss} --times 1 --dataset 'ogbn-arxiv' --batchsize 2048 --max_duration 60 --kmeans_device 'cpu' --kmeans_batch -1 --hidden '1024,256' --size '10,10' --wt 20 --wl 5 --tau 0.9 --ns 0.1 --lr 0.01 --epochs 400 --wd 0 --dropout 0
  ```
* reddit
  ```
  sss="$(date +%Y-%m-%d-%H-%s)/"
  mkdir ./log
  mkdir ./log/reddit
  mkdir ./log/reddit/${sss}
  python ./train_sage.py --verbose True --log True --log_file './log/reddit/'${sss} --times 1 --dataset 'Reddit' --batchsize 2048 --max_duration 60 --kmeans_device 'cpu' --kmeans_batch -1 --hidden '1024,256' --size '10,10' --wt 20 --wl 5 --tau 0.5 --ns 0.5 --lr 0.01 --epochs 400 --wd 0 --dropout 0
  ```
* ogbn-products
  ```
  sss="$(date +%Y-%m-%d-%H-%s)/"
  mkdir ./log
  mkdir ./log/ogbn_products
  mkdir ./log/ogbn_products/${sss}
  python ./train_sage.py --verbose True --log True --log_file './log/ogbn_products/'${sss} --times 1 --dataset 'ogbn-products' --batchsize 2048 --max_duration 60 --kmeans_device 'cuda' --kmeans_batch 300000 --hidden '1024,1024,256' --size '10,10,10' --wt 20 --wl 4 --tau 0.9 --ns 0.1 --lr 0.01 --epochs 400 --wd 0 --dropout 0
  ```




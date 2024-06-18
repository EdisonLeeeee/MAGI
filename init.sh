
# pip install torch_geometric==2.0.4
# pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# python -c "import torch; print(torch.__version__)"
# set -x
# pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
# pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
# pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
# pip install scikit-learn scikit-learn-intelex
# pip install ogb==1.3.6
# pip install munkres
# pip install matplotlib






# sss="$(date +%Y-%m-%d-%H-%s)/"
# mkdir ./log
# mkdir ./log/cora
# mkdir ./log/cora/${sss}
# python ./train_gcn.py --verbose True --log True --log_file './log/cora/'${sss} --times 10 --dataset 'Cora' --hidden '512' --wt 100 --wl 2 --tau 0.3 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3


# sss="$(date +%Y-%m-%d-%H-%s)/"
# mkdir ./log
# mkdir ./log/citeseer
# mkdir ./log/citeseer/${sss}
# python ./train_gcn.py --verbose True --log False --log_file './log/citeseer/'${sss} --times 10 --dataset 'Citeseer' --hidden '1024,512' --wt 100 --wl 3 --tau 0.9 --ns 0.5 --lr 0.0001 --epochs 400 --wd 5e-4


# sss="$(date +%Y-%m-%d-%H-%s)/"
# mkdir ./log
# mkdir ./log/photo
# mkdir ./log/photo/${sss}
# python ./train_gcn.py --verbose True --log True --log_file './log/photo/'${sss} --times 10 --dataset 'Photo' --hidden '512' --wt 100 --wl 3 --tau 0.5 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3


# sss="$(date +%Y-%m-%d-%H-%s)/"
# mkdir ./log
# mkdir ./log/computers
# mkdir ./log/computers/${sss}
# python ./train_gcn.py --verbose True --log True --log_file './log/computers/'${sss} --times 10 --dataset 'Computers' --hidden '1024,512' --wt 100 --wl 3 --tau 0.9 --ns 0.1 --lr 0.0005 --epochs 400 --wd 1e-3


# sss="$(date +%Y-%m-%d-%H-%s)/"
# mkdir ./log
# mkdir ./log/ogbn_arxiv
# mkdir ./log/ogbn_arxiv/${sss}
# python ./train_sage.py --verbose True --log True --log_file './log/ogbn_arxiv/'${sss} --times 1 --dataset 'ogbn-arxiv' --batchsize 2048 --max_duration 60 --kmeans_device 'cpu' --kmeans_batch -1 --hidden '1024,256' --size '10,10' --wt 20 --wl 5 --tau 0.9 --ns 0.1 --lr 0.01 --epochs 400 --wd 0 --dropout 0


# sss="$(date +%Y-%m-%d-%H-%s)/"
# mkdir ./log
# mkdir ./log/reddit
# mkdir ./log/reddit/${sss}
# python ./train_sage.py --verbose True --log True --log_file './log/reddit/'${sss} --times 1 --dataset 'Reddit' --batchsize 2048 --max_duration 60 --kmeans_device 'cpu' --kmeans_batch -1 --hidden '1024,256' --size '10,10' --wt 20 --wl 5 --tau 0.5 --ns 0.5 --lr 0.01 --epochs 400 --wd 0 --dropout 0


# sss="$(date +%Y-%m-%d-%H-%s)/"
# mkdir ./log
# mkdir ./log/ogbn_products
# mkdir ./log/ogbn_products/${sss}
# python ./train_sage.py --verbose True --log True --log_file './log/ogbn_products/'${sss} --times 1 --dataset 'ogbn-products' --batchsize 2048 --max_duration 60 --kmeans_device 'cuda' --kmeans_batch 300000 --hidden '1024,1024,256' --size '10,10,10' --wt 20 --wl 4 --tau 0.9 --ns 0.1 --lr 0.01 --epochs 400 --wd 0 --dropout 0



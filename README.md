# AuditVotes-Certified-Robustness
AuditVotes Certified Robustness Source Code for the paper: "AuditVotes: A Framework Towards More Deployable Certified Robustness for Graph Neural Networks"


### Datasets are obtained from:

https://github.com/gasteigerjo/ppnp (Node classification)  

### Our Code is adapted from:
```
@inproceedings{bojchevski_sparsesmoothing_2020,
title = {Efficient Robustness Certificates for Discrete Data: Sparsity-Aware Randomized Smoothing for Graphs, Images and More},
author = {Bojchevski, Aleksandar and Klicpera, Johannes and G{\"u}nnemann, Stephan},
booktitle = {Proceedings of the 37th International Conference on Machine Learning},
pages = {1003--1013},
year = {2020},
url={https://github.com/abojchevski/sparse_smoothing}
}
```

### Environment setup

```bash
conda env create -f ./Environment/dgl.yml
conda activate dgl
pip install -r ./Environment/dgl.txt
```
or with specific dir:
```bash
conda env create -f ./Environment/dgl.yml -p /home/xxx/dgl
conda activate /home/xxx/dgl
pip install -r ./Environment/dgl.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.

### Run the AuditVotes
All the training, smoothing, and certifying processes are in main.py.  
For example, to run GCN+SimAug+Conf on the Cora-ML dataset:
```bash
main.py -dataset 'cora_ml' -pf_plus_adj 0.2 -pf_minus_adj 0.6 -certify_type 'r_a' -certify_mode 'WithDetect' -filter 'Conf' -model 'GCN' -augmenter 'SimAug'
```




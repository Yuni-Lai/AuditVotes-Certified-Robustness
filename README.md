# AuditVotes-Certified-Robustness
AuditVotes Certified Robustness Source Code for the paper: "AuditVotes: A Framework Towards More Deployable Certified Robustness for Graph Neural Networks"



### Datasets are obtained from:

https://github.com/gasteigerjo/ppnp (Node classification)  
https://github.com/XiaFire/GNNCERT (Graph classification)

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
and
```
@inproceedings{
xia2024gnncert,
title={GNNCert: Deterministic Certification of Graph Neural Networks against Adversarial Perturbations},
author={zaishuo xia and Han Yang and Binghui Wang and Jinyuan Jia},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://github.com/XiaFire/GNNCERT}
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




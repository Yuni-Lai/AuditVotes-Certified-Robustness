# AuditVotes-Certified-Robustness
AuditVotes Certified Robustness Source Code for the paper: "AuditVotes: Elevating Provable Defense for GNNs with Efficient Augmentation and Conditional Smoothing"


### Datasets are obtained from:

https://github.com/gasteigerjo/ppnp (Node classification)  
We put them in the ./Data folder.
```bash
/Data
```
### Environment setup

```bash
conda env create -f ./Environment/dgl.yml
conda activate dgl
pip install -r ./Environment/dgl.txt
```
or with a specific dir:
```bash
conda env create -f ./Environment/dgl.yml -p [dir]
conda activate [dir]
pip install -r ./Environment/dgl.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.

### Run the AuditVotes
All the training, smoothing, and certifying processes are included in ./NodeClassify_SparseSmooth/main.py.  

For example, to reproduce the results of Table 3 (certify $r_d$ with $p_+=0.0$ and $p_-=0.8$):

Baseline SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.0 -pf_minus_adj 0.8 -certify_type 'r_d' -certify_mode 'Vanilla' -model 'GCN' -augmenter ''
```
Apply AuditVotes(SimAug) to SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.0 -pf_minus_adj 0.8 -certify_type 'r_d' -certify_mode 'Vanilla' -model 'GCN' -augmenter 'SimAug'
```
Apply AuditVotes(SimAug+Conf) to SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.0 -pf_minus_adj 0.8 -certify_type 'r_d' -certify_mode 'WithDetect' -filter 'Conf' -model 'GCN' -augmenter 'SimAug'
```
Apply AuditVotes(FAEAug+Conf) to SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.0 -pf_minus_adj 0.8 -certify_type 'r_d' -certify_mode 'WithDetect' -filter 'Conf' -model 'GCN' -augmenter 'FAEAug'
```
Apply AuditVotes(JacAug+Conf) to SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.0 -pf_minus_adj 0.8 -certify_type 'r_a' -certify_mode 'WithDetect' -filter 'Conf' -model 'GCN' -augmenter 'JacAug'
```
To reproduce the results of Table 4 (certify $r_a$ with $p_+=0.2$ and $p_-=0.6$):

Baseline SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.2 -pf_minus_adj 0.6 -certify_type 'r_a' -certify_mode 'Vanilla' -model 'GCN' -augmenter ''
```
Apply AuditVotes(SimAug) to SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.2 -pf_minus_adj 0.6 -certify_type 'r_a' -certify_mode 'Vanilla' -model 'GCN' -augmenter 'SimAug'
```
Apply AuditVotes(SimAug+Conf) to SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.2 -pf_minus_adj 0.6 -certify_type 'r_a' -certify_mode 'WithDetect' -filter 'Conf' -model 'GCN' -augmenter 'SimAug'
```
Apply AuditVotes(FAEAug+Conf) to SparseSmooth on the Citeseer dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'citeseer' -pf_plus_adj 0.2 -pf_minus_adj 0.6 -certify_type 'r_a' -certify_mode 'WithDetect' -filter 'Conf' -model 'GCN' -augmenter 'FAEAug'
```

For other schemes, refer to README.md in corresponding folders.



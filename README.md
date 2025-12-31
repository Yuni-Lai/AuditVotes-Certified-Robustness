# AuditVotes-Certified-Robustness
AuditVotes Certified Robustness Source Code for the paper: "AuditVotes: Elevating Provable Defense for GNNs with Efficient Augmentation and Conditional Smoothing"


### Datasets are obtained from:

https://github.com/gasteigerjo/ppnp (Node classification)  



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
All the training, smoothing, and certifying processes are in ./NodeClassify_SparseSmooth/main.py.  
For example, Apply AuditVotes(SimAug+Conf) to SparseSmooth on the Cora-ML dataset:
```bash
cd ./NodeClassify_SparseSmooth
python main.py -dataset 'cora_ml' -pf_plus_adj 0.2 -pf_minus_adj 0.6 -certify_type 'r_a' -certify_mode 'WithDetect' -filter 'Conf' -model 'GCN' -augmenter 'SimAug'
```

For other schemes, refer to README.md in corresponding folders.



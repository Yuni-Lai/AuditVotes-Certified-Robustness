#!/usr/bin/env bash
#conda activate dgl
#cd ~/MyProjects/CertifiyWithfilter/NodeClassify_SparseAuditVotes

#nohup bash run.sh > ./run.log 2>&1 &
#datset=['cora_ml', 'citeseer', 'pubmed']
#mode=['Vanilla', 'WithDetect']


#-------
#mode='WithDetect'
#model='GCN'

#for dataset in 'citeseer' 'cora_ml' #'citeseer' 'cora_ml'
#do
#  for filter in 'BWGNN_Conf' 'BWGNN' #'Conf' 'Homo' 'GCN_Conf'
#  do
#      echo "The current program (dataset,filter) is: ${dataset}, ${filter}"
#      nohup python -u main.py -dataset $dataset -pf_minus_adj 0.5 -filter $filter -model $model -certify_mode $mode -gpuID 7 -seed $seed > ./results_${dataset}_${model}/${mode}${filter}/run_0.5.log 2>&1 &
#      nohup python -u main.py -dataset $dataset -pf_minus_adj 0.8 -filter $filter -model $model -certify_mode $mode -gpuID 8 -seed $seed > ./results_${dataset}_${model}/${mode}${filter}/run_0.8.log 2>&1 &
#      wait
#  done
#done


#-------
seed=2021
#mode='WithDetect' #['Vanilla', 'WithDetect']
filter='Conf'
certify_type='r_a'
#  for model in 'GCN' 'JacGCN' 'FAugGCN' 'SAugGCN' #'GCN' 'JacGCN' 'FAugGCN' 'SAugGCN'

echo "Certifying r_a"

for dataset in 'pubmed' #'citeseer' 'cora_ml' #'citeseer' 'cora_ml' 'pubmed'
do
  for pf_plus in 0.2 #0.0 0.1 0.2 0.3 0.4
  do
    for pf_minus in 0.6 #0.0 0.2 0.3 0.4 0.5 0.6 #0.4 0.5 0.6 0.7 0.8
    do
      echo "The current program (dataset,pf_plus,pf_minus) is: ${dataset}, ${pf_plus}, ${pf_minus}"
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'GCN' -gpuID 5 -seed $seed > ./results_${dataset}_GCN/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'JacGCN' -gpuID 6 -seed $seed > ./results_${dataset}_GCNJaccard_Aug/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'FAugGCN' -gpuID 7 -seed $seed > ./results_${dataset}_FAugGCN/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'SAugGCN' -gpuID 8 -seed $seed > ./results_${dataset}_SAugGCN/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      wait
    done
  done
done
#
for dataset in 'pubmed' # 'citeseer' 'cora_ml' #'citeseer' 'cora_ml' 'pubmed'
do
  for pf_plus in 0.2 # 0.1 0.2 0.3 0.4 #0.0 0.1 0.2 0.3 0.4
  do
    for pf_minus in 0.6 #0.0 0.2 0.3 0.4 0.5 0.6 #0.4 0.5 0.6 0.7 0.8
    do
      echo "The current program (dataset,pf_plus,pf_minus) is: ${dataset}, ${pf_plus}, ${pf_minus}"
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'GCN' -gpuID 5 -seed $seed > ./results_${dataset}_GCN/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'JacGCN' -gpuID 6 -seed $seed > ./results_${dataset}_GCNJaccard_Aug/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'FAugGCN' -gpuID 7 -seed $seed > ./results_${dataset}_FAugGCN/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'SAugGCN' -gpuID 8 -seed $seed > ./results_${dataset}_SAugGCN/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
      wait
    done
  done
done

#----------------------------------------------------------
#echo "Certifying r_d"
#certify_type='r_d'
##  for model in 'GCN' 'JacGCN' 'FAugGCN' 'SAugGCN' #'GCN' 'JacGCN' 'FAugGCN' 'SAugGCN'
#
#for dataset in 'pubmed' # 'citeseer' 'cora_ml' 'pubmed'
#do
#  for pf_plus in 0.0 0.01 #0.0 0.1 0.2 0.3 0.4
#  do
#    for pf_minus in 0.4 0.5 0.6 0.7 0.8 #0.0 0.2 0.3 0.4 0.5 0.6 #0.4 0.5 0.6 0.7 0.8
#    do
#      echo "The current program (dataset,pf_plus,pf_minus) is: ${dataset}, ${pf_plus}, ${pf_minus}"
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'GCN' -gpuID 5 -seed $seed > ./results_${dataset}_GCN/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'JacGCN' -gpuID 6 -seed $seed > ./results_${dataset}_GCNJaccard_Aug/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'FAugGCN' -gpuID 7 -seed $seed > ./results_${dataset}_FAugGCN/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'Vanilla' -filter $filter -model 'SAugGCN' -gpuID 8 -seed $seed > ./results_${dataset}_SAugGCN/run_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      wait
#    done
#  done
#done
#
#for dataset in 'pubmed'  # 'citeseer' 'cora_ml' 'pubmed'
#do
#  for pf_plus in 0.0 0.01 # 0.1 0.2 0.3 0.4 #0.0 0.1 0.2 0.3 0.4
#  do
#    for pf_minus in 0.4 0.5 0.6 0.7 0.8 #0.0 0.2 0.3 0.4 0.5 0.6 #0.4 0.5 0.6 0.7 0.8
#    do
#      echo "The current program (dataset,pf_plus,pf_minus) is: ${dataset}, ${pf_plus}, ${pf_minus}"
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'GCN' -gpuID 5 -seed $seed > ./results_${dataset}_GCN/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'JacGCN' -gpuID 6 -seed $seed > ./results_${dataset}_GCNJaccard_Aug/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'FAugGCN' -gpuID 7 -seed $seed > ./results_${dataset}_FAugGCN/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      nohup python -u main.py -dataset $dataset -pf_plus_adj $pf_plus -pf_minus_adj $pf_minus -certify_type $certify_type -certify_mode 'WithDetect' -filter $filter -model 'SAugGCN' -gpuID 8 -seed $seed > ./results_${dataset}_SAugGCN/run_conf_adj_${pf_plus}_${pf_minus}.log 2>&1 &
#      wait
#    done
#  done
#done
#echo "Proccess Finished!"

# CertifiyWithDetector
AuditVotes apply to GNNCert, enhancing certified robustness with augmentation

### ataset obtained from:
https://github.com/gasteigerjo/ppnp

### Code adapted from:
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

### Usage
Applying AuditVotes(SimAug) to GNNCert: 

```python
python main.py --dataset 'cora_ml' --model 'SAugGCN' --certify_mode 'Vanilla' --Ts 20
```






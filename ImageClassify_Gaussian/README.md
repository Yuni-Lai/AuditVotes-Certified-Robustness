# Certify Image Classification with AuditVotes

Apply the AuditVotes (Conf) to Gaussian to certify the robustness of image classification models.

###  Code adapted from:
Gaussian (Certified Adversarial Robustness via Randomized Smoothing):
``` bash
@inproceedings{cohen2019certified,
  title={Certified adversarial robustness via randomized smoothing},
  author={Cohen, Jeremy and Rosenfeld, Elan and Kolter, Zico},
  booktitle={international conference on machine learning},
  pages={1310--1320},
  year={2019},
  organization={PMLR}
}
```

### Usage

To certify an image classification model using AuditVotes (Conf) with Gaussian smoothing, run:

``` bash
cd ./code/
python training.py (The same procedure as Gaussian training)
python predict.py (The same procedure as Gaussian prediction)
python certify.py --certify_mode 'WithDetect' --confs 0.9 
```

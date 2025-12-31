# evaluate a smoothed classifier on a dataset
import argparse
import os

import pandas as pd

# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core_1 import Smooth
from time import time
import torch
import datetime
import random
import numpy as np
from tqdm.autonotebook import tqdm
from architectures import get_architecture
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
MEDIUM_SIZE = 25
BIGGER_SIZE = 27
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--seed", type=int, default=1000, help="ranodm seed")
parser.add_argument("--dataset", type=str, default="cifar10", choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str, default="../models/{dataset}/resnet110/noise_{sigma}/checkpoint.pth.tar",
                    help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", default=0.25, type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str,default=" ", help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--test_num", type=int, default=100, help="number of testing")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--confs",type=float, default=0.9, help= "confidence score")
parser.add_argument('--certify_mode', type=str, default='WithDetect', choices=['Vanilla', 'WithDetect'])
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    args.base_classifier = args.base_classifier.format(dataset=args.dataset, sigma=args.sigma)
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, args.confs, args.certify_mode)
    # prepare output file
    args.outfile = f'./results_{args.dataset}/resnet110_noise_{args.sigma}'
    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)
    if args.certify_mode == 'WithDetect':
        output_dir = f'{args.outfile}/Conf_{args.confs}'
    else:
        output_dir = f'{args.outfile}/Vanilla'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f = open(f'{output_dir}/certified_acc', 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    correct_radius=[]
    before_time = time()
    random.seed(args.seed)
    # sampled_index=random.sample(range(len(dataset)),args.test_num)
    for i in tqdm(range(len(dataset))):#len(dataset)

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()

        # certify the prediction of g around x

        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)

        correct = int(prediction == label)
        if correct:
            correct_radius.append(radius)

        print("{}\t{}\t{}\t{:.3}\t{}".format(
            i, label, prediction, radius, correct), file=f, flush=True)
    f.close()
    after_time = time()
    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

    certified_acc=[]
    rho_list = [*np.arange(0,2,0.1)]
    for i in rho_list:
        certified_acc.append((np.array(correct_radius)>i).sum()/args.test_num)

    pre_df = pd.DataFrame({'rho': rho_list, 'certified accuracy': certified_acc})
    pre_df.to_pickle(f'{output_dir}/certified_results.pkl')

    df = {'rho': rho_list, 'certified accuracy': certified_acc}
    plt.figure(constrained_layout=True)
    sns.lineplot(x="rho", y="certified accuracy", data=df)
    plt.ylim(0, 1)
    plt.xlabel(r'$\rho$')
    plt.title(fr"{args.dataset}, $\sigma$={args.sigma}, $N$={args.N}")
    plt.savefig(f'{output_dir}/certified_acc_curve.pdf',
                    dpi=300)
    print(f'Save result to {output_dir}/certified_acc_curve.pdf')
    plt.show()





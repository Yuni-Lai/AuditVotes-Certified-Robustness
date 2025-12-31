import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import argparse

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


parser = argparse.ArgumentParser(description='visualize the merged certified_acc curves')
parser.add_argument("--input_dir", type=str,default="./results_cifar10/resnet110_noise_0.25", help="input directory")
parser.add_argument("--sigma", default=0.25, type=float, help="noise hyperparameter")
parser.add_argument("--dataset", type=str, default="cifar10", help="which dataset")
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
args = parser.parse_args()


conf_levels = [ "Conf_0.5", "Conf_0.6", "Conf_0.7", "Conf_0.8", "Conf_0.9"]#"Vanilla",
file_paths = [f'{args.input_dir}/{conf}/certified_results.pkl' for conf in conf_levels]
name_map = {'cifar10': 'CIFAR-10', 'mnist': 'MNIST'}

dfs = [pd.read_pickle(file_path) for file_path in file_paths]
for i,df in enumerate(dfs):
    df["models"] = conf_levels[i]
df_combine = pd.concat(dfs, ignore_index=True)
df_combine['models'] = df_combine['models'].str.replace("_", ">")
plt.figure(constrained_layout=True, figsize=(8, 6))
plt.title(fr"{name_map[args.dataset]}, $\sigma$={args.sigma}, $N$={args.N}")
sns.lineplot(x="rho", y="certified accuracy", hue='models', data=df_combine, linewidth=2,palette='hls')#palette=['#8f8787', '#ffcc5c', '#88d8b0', '#ff6f69']
plt.xlabel(r'$l_2$-radius', fontsize=28)
plt.savefig(f'./merged_certified_acc_curve.pdf',
                    dpi=300, bbox_inches='tight')
print(f'Save result to {args.input_dir}/merged_certified_acc_curve.pdf')
plt.show()

df_combine=df_combine.round({'rho': 2})
print(df_combine[df_combine.loc[:, 'rho'].isin([0.0,0.3,0.5,0.7,1.0])])

table = df_combine.pivot_table(index=['models'], columns='rho',
                           values='certified accuracy', sort=False)
print(table.loc[:, [0,0.3,0.5,0.7,1.0]])
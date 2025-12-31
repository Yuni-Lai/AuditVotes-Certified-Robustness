import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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

name_map = {'cora_ml': 'Cora-ML', 'citeseer': 'Citeseer','pubmed':'PubMed'}

def analyze_result(analysis_data,output_dir,args):
    df = pd.DataFrame(analysis_data)
    df["correct_pred"]=df["class_pred"]==df["class_label"]
    # plt.figure(constrained_layout=True, figsize=(8, 6))
    # plt.title("histogram of anomaly scores by classification result")
    # # df.groupby('correct_pred')['anomaly_score'].hist(bins=100)
    # sns.histplot(x='anomaly_score', hue='correct_pred',multiple="dodge",palette = "Set1",bins=50, data = df)
    # plt.legend(['correctly classified', 'incorrectly classified'],loc='upper right', fancybox=True, framealpha=0.5)
    # plt.savefig(output_dir + 'analysis1.pdf', dpi=300, bbox_inches='tight')
    # print(f'Save result to {output_dir}analysis1.pdf')
    # plt.show()

    # plt.figure(constrained_layout=True, figsize=(8, 6))
    # plt.title("violin plots of detection results by classification result")
    # sns.violinplot(x='correct_pred', y='anomaly_score',palette = "Set1", data = df)
    # # plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
    # plt.savefig(output_dir + 'analysis2.pdf', dpi=300, bbox_inches='tight')
    # print(f'Save result to {output_dir}analysis2.pdf')
    # plt.show()

    # plt.figure(constrained_layout=True, figsize=(8, 6))
    # plt.title("Node classification correctness group by detection result")
    # sns.histplot(x='anomaly_pred', hue='correct_pred',multiple="dodge",bins=10,palette = "Set1", data = df)
    # plt.xlabel('anomaly predction')
    # # plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
    # plt.savefig(output_dir + 'analysis3.pdf', dpi=300, bbox_inches='tight')
    # print(f'Save result to {output_dir}analysis3.pdf')
    # plt.show()

    df1={'classification accuracy':[df.groupby('anomaly_pred')['correct_pred'].mean()[0],df.groupby('anomaly_pred')['correct_pred'].mean()[1]],'anomaly predction':['Normal','Anomaly']}
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{name_map[args.dataset]},$(T_s,T_f)$=({args.Ts},{args.Tf})")
    ax=sns.barplot(x='anomaly predction', y='classification accuracy', width=0.5,palette="Set1", data=df1)
    ax.bar_label(ax.containers[0], fontsize=22);
    plt.ylabel('node classification accuracy')
    plt.ylim(0,1.0)
    # plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
    plt.savefig(output_dir + 'analysis4.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}analysis4.pdf')
    plt.show()

def certified_curve(df,output_dir,args):
    # plt.figure(constrained_layout=True, figsize=(8, 6))
    # sns.lineplot(x="rd", y="certified ratio", data=df)
    # plt.title(fr"{name_map[args.dataset]},$(T_s,T_f)$=({args.Ts},{args.Tf})")
    # plt.savefig(output_dir + 'certifiedratio.pdf', dpi=300, bbox_inches='tight')
    # print(f'Save result to {output_dir}certifiedratio.pdf')
    # plt.show()

    plt.figure(constrained_layout=True, figsize=(8, 6))
    sns.lineplot(x="rd", y="certified accuracy", data=df)
    plt.title(fr"{name_map[args.dataset]},$(T_s,T_f)$=({args.Ts},{args.Tf})")
    plt.savefig(output_dir + 'certifiedaccuracy.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}certifiedaccuracy.pdf')
    plt.show()

def merged_certified_curve(df_combine,hue,output_dir,args):
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{name_map[args.dataset]},$(T_s,T_f)$=({args.Ts},{args.Tf})")
    sns.lineplot(x="rd", y="certified accuracy",hue=hue, data=df_combine,linewidth = 3)
    plt.ylim(0,1)
    plt.legend(loc='upper right', fancybox=True,framealpha=0.5, title=hue)  #
    plt.savefig(output_dir + f'merged_{hue}_certaccuracy.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}merged_{hue}_certaccuracy.pdf')
    plt.show()

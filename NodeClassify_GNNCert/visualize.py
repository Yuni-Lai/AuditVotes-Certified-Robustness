import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import pickle
import argparse
import os.path
import os
from certify import model_map

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
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rcParams['legend.title_fontsize'] = MEDIUM_SIZE
palette = ["#ffaaa5","#adcbe3", "#5779C1"]
font = {'family': 'serif',
        'color': 'darkblue',
        'weight': 'normal',
        'size': 5}

#-----string mapping----
name_map = {'cora_ml': 'Cora-ML', 'citeseer': 'Citeseer','pubmed':'PubMed'}

def filter_map(detector):
    filter_mapdict = {'Conf': 'Confidence', 'Homo': 'Homophily', 'NSP': 'NSP', 'Prox1': '-Prox1', 'Prox2': '-Prox2', 'JSD':'-JSDivergence','Others': 'Quality'}
    if detector in ['Conf', 'Homo', 'Prox1', 'Prox2', 'JSD', 'NSP']:
        return filter_mapdict[detector]
    else:
        return filter_mapdict['Others']

# def direction_map(detector):
#     if detector in ['Prox1', 'Prox2', 'JSD']: #Lower is better
#         return ['High','Low']
#     else: #Higher is better
#         return ['Low','High']

def thre_map(args):
    if args.detector == 'Conf':
        return f'Threshold: Confidence>{1-args.conf_thre}'
    elif args.detector == 'Homo':
        return f'Threshold: Homophily>{1-args.homo_thre}'
    elif args.detector == 'Prox1':
        return f'Threshold: -Prox1>{-args.prox_thre}'
    elif args.detector == 'Prox2':
        return f'Threshold: -Prox2>{-args.prox_thre}'
    elif args.detector == 'JSD':
        return f'Threshold: -JSD>{-args.jsd_thre}'
    elif args.detector == 'NSP':
        return f'Threshold: NSP>{np.round(1-args.nsp_thre,2)}'
    else:
        return f'Threshold: Upper {np.round((1-args.quality_thre)*100,2)}%'
#-----------------------

def get_type_df(args,type,mode="detector"):
    if mode=="detector_name":
        out_dir_type = f'./results_{args.dataset}_{args.model}/{type}/Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}/'
        f = open(f'{out_dir_type}/certify_result.pkl', 'rb')
        df = pickle.load(f)
        f.close()
    else:
        out_dir_type = f'./results_{args.dataset}_{type}/Vanilla/Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}/'
        f = open(f'{out_dir_type}/certify_result.pkl', 'rb')
        df = pickle.load(f)
        f.close()
        # df["certify_mode"] = f'{type}-Vanilla'
    return df,out_dir_type

def get_combine_df(args,types):
    for i,type in enumerate(types):
        if type in ["Vanilla","WithDetectConf", "WithDetectHomo"]:
            df,out_dir_type=get_type_df(args, type, mode="detector_name")
        else:
            df,out_dir_type=get_type_df(args, type, mode="model_name")
        if i==0:
            df_combine = df
        else:
            df_combine = pd.concat([df_combine, df], ignore_index=True)
        if i == len(types)-1:
            f = open(f'{out_dir_type}/smoothing_result.pkl', 'rb')
            votes_vanilla = pickle.load(f) #for pA distribution visualization
            f.close()
    return df_combine, votes_vanilla


def analyze_result(analysis_data,output_dir,args):
    df = pd.DataFrame(analysis_data)
    df = df.sample(n=30000)
    df["correct_pred"] = df["class_pred"]==df["class_label"]

    #distribution
    df['quality_score']=1-df['anomaly_score']
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(f"{filter_map(args.detector)} distribution")
    plt.xlabel(f'{filter_map(args.detector)}')
    # df.groupby('correct_pred')['anomaly_score'].hist(bins=100)
    #sns.histplot(x='anomaly_score', hue='correct_pred',multiple="dodge",stat="count",palette = "Set1",bins=50, data = df)
    sns.histplot(data=df[df['correct_pred']==True], x="quality_score", color="skyblue", stat="density", label="classified correctly", kde=True, kde_kws={"bw_adjust":2},bins=30)
    sns.histplot(data=df[df['correct_pred']==False], x="quality_score", color="red", stat="density", label="classified incorrectly", kde=True, kde_kws={"bw_adjust":2},bins=30)
    plt.legend(['correctly classified', 'incorrectly classified'], loc='upper right', fancybox=True, framealpha=0.5)
    plt.savefig(output_dir + f'analysis1.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}analysis1.pdf')
    plt.show()

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
    try:
        df1 = {'classification accuracy': [df.groupby('anomaly_pred')['correct_pred'].mean()[1],
                                           df.groupby('anomaly_pred')['correct_pred'].mean()[0]],
               'anomaly predction': [f'Low {filter_map(args.detector)}', f'High {filter_map(args.detector)}']}

        plt.figure(constrained_layout=True, figsize=(8, 6))
        plt.title(fr"{name_map[args.dataset]},{thre_map(args)}")
        ax=sns.barplot(x='anomaly predction', y='classification accuracy', width=0.4,palette=palette, data=df1)#'Set1'
        ax.bar_label(ax.containers[0], fontsize=22);
        plt.ylabel('node classification accuracy')
        plt.ylim(0,1.0)
        plt.xlabel(f'${args.certify_type}$', fontsize=28)
        # plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
        plt.savefig(output_dir + f'analysis4.pdf', dpi=300, bbox_inches='tight')
        print(f'Save result to {output_dir}analysis4.pdf')
        plt.show()
    except:
        print('errors in df1: df.groupby(anomaly_pred)[correct_pred].mean()[0]]')
        pass

    print('Anomaly ratio:',sum(analysis_data['anomaly_pred'])/len(analysis_data['anomaly_pred']))

def certified_curve(df,output_dir,args):
    # plt.figure(constrained_layout=True, figsize=(8, 6))
    # sns.lineplot(x=args.certify_type, y="certified ratio", data=df)
    # plt.title(fr"{name_map[args.dataset]},$(p_+,p_-)$=({args.pf_plus_adj},{args.pf_minus_adj}),N={args.N}")
    # plt.savefig(output_dir + 'certifiedratio.pdf', dpi=300, bbox_inches='tight')
    # print(f'Save result to {output_dir}certifiedratio.pdf')
    # plt.show()

    plt.figure(constrained_layout=True, figsize=(8, 6))
    sns.lineplot(x=args.certify_type, y="certified accuracy", data=df)
    plt.xlabel(f'${args.certify_type}$', fontsize=28)
    plt.title(fr"{name_map[args.dataset]},$(T_s,T_f)$=({args.Ts},{args.Tf})")
    plt.savefig(output_dir + f'certifiedaccuracy_{args.certify_type}.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}certifiedaccuracy_{args.certify_type}.pdf')
    plt.show()

def merged_certified_curve(df_combine,output_dir,args):
    # type_map={'GCN-Vanilla':'GCN','GCNJaccard_Aug-Vanilla':'GCN+JacAug','+FilterConf':'GCN+JacAug+Conf'}
    # df_combine=df_combine.replace(type_map)
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{name_map[args.dataset]},$(T_s,T_f)$=({args.Ts},{args.Tf})")
    sns.lineplot(x=args.certify_type, y="certified accuracy",hue='certify_mode', palette=['#8f8787', '#ffcc5c','#88d8b0','#ff6f69','#92a8d1'], data=df_combine,linewidth = 3)
    plt.xlabel(f'${args.certify_type}$', fontsize=28)
    plt.ylim(0.4,0.8)
    # plt.xlim(0,8)
    plt.legend(loc='upper right', fancybox=True,framealpha=0.5)  #
    plt.savefig(output_dir + f'merged_certaccuracy_{args.certify_type}.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}merged_certaccuracy_{args.certify_type}.pdf')
    plt.show()

def pA_distribution(votes_vanilla,votes,labels, args):
    labels=labels.cpu().numpy()
    gap = np.sort(votes,axis=1)[:,-1] - np.sort(votes,axis=1)[:,-2]
    df = pd.DataFrame({'nodes':range(votes.shape[0]),'gap':gap})

    gap_vanilla = np.sort(votes_vanilla, axis=1)[:, -1] - np.sort(votes_vanilla, axis=1)[:, -2]
    df_vanilla = pd.DataFrame({'nodes': range(votes_vanilla.shape[0]), 'gap': gap_vanilla})

    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{name_map[args.dataset]}, gap distribution")
    sns.histplot(data=df_vanilla, x="gap", color="dimgrey", label='Vanilla', stat="density", kde=True, bins=10)
    sns.histplot(data=df, x="gap", color="dodgerblue",label=f'+Filter-{args.detector}', stat="density",kde=True, bins=10)
    plt.xlabel('gap between top-1 & top-2 class', fontsize=28)
    plt.legend(loc='upper left', fancybox=True, framealpha=0.5)  #
    plt.savefig(args.output_dir + 'pA_distribution.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {args.output_dir}pA_distribution.pdf')
    plt.show()

    return None

def load_data(dir, file):
    f = open(dir + file, 'rb')
    df = pickle.load(f)
    f.close()
    return df

def clean_certify_figure(args,combine_list):
    # color_code = ['#740001','#ae0001' ,'#eeba30','#d3a625','#000000','#0e1a40','#222f5b','#5d5d5d','#946b2d']
    # color_code = ["#00aedb","#f47835","#d41243","#8ec127","#5d5d5d","#a200ff","#ffbf00","#222f5b","#946b2d"]
    # color_code = ['#e1f7d5','#ffbdbd',"#c9c9ff","#f1cbff","#ff71ce","#01cdfe","#05ffa1","#b967ff","#fffb96"]
    # color_code = ['#740001','#ae0001' ,'#eeba30','#d3a625',"#5d5d5d","#a200ff","#ffbf00","#222f5b","#946b2d"]
    # color_code = ['#740001','#ae0001' ,'#eeba30','#d3a625','#d11141','#00b159','#00aedb','#f37735','#ffc425']

    root_dir = './results_{dataset}_{model}/{certify_mode}/{paras}'
    file_dir = f'/certify_result.pkl'
    output_dir = f'./combined_figures/{args.dataset}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #load data----------------certify_mode='Vanilla'
    df_combine1 = pd.DataFrame()
    for model in ['GCN', 'GCNJaccard_Aug', 'FAugGCN', 'SAugGCN']:
        for pp in combine_list:
            pp = pp + f'_{args.n_smoothing}'
            print(pp)
            try:
                df1 = pd.DataFrame(load_data(root_dir.format(paras=pp,dataset=args.dataset,model=model,certify_mode='Vanilla'), file_dir))
                df1['clean_accuracy'] = df1[df1[args.certify_type] == 0]['certified accuracy'][0]
                df1['parameters'] = f"$p_+$:{pp.split('_')[0]},$p_-$:{pp.split('_')[2]}"
                df1['model'] = model_map[model]
                df1['condition'] = 'None'
                df_combine1 = pd.concat([df_combine1, df1], ignore_index=True)
            except:
                print(f"not found {model},{pp}, vanilla")
                pass


    # load data----------------certify_mode='WithDetectConf'
    df_combine2 = pd.DataFrame()
    for model in ['GCN', 'GCNJaccard_Aug', 'FAugGCN','SAugGCN']:
        for pp in combine_list:
            pp = pp + f'_{args.n_smoothing}'
            print(pp)
            try:
                df2 = pd.DataFrame(
                    load_data(root_dir.format(paras=pp, dataset=args.dataset, model=model, certify_mode='WithDetectConf'),
                              file_dir))

                df2['clean_accuracy'] = df2[df2[args.certify_type] == 0]['certified accuracy'][0]
                df2['parameters'] = f"$p_+$:{pp.split('_')[0]},$p_-$:{pp.split('_')[2]}"
                df2['model'] = model_map[model]
                df2['condition'] = '+Conf'
                df_combine2 = pd.concat([df_combine2, df2], ignore_index=True)
            except:
                print(f"not found {model},{pp}, withdetect")
                pass

    if args.include_both:
        df_all = pd.concat([df_combine1,df_combine2]).dropna(axis=1).reset_index()
    elif args.certify_mode == 'Vanilla':
        df_all = df_combine1
    else:
        df_all = df_combine2

    df_r_all = df_all.loc[df_all[args.certify_type] == args.r]
    df_r_all = df_r_all.loc[df_r_all['clean_accuracy'] >= MLP_acc[args.dataset]]
    params = list(df_r_all.loc[df_r_all['certified accuracy'] >= 0.01]['parameters'])
    df_r_all = df_r_all.loc[df_r_all['parameters'].isin(params)]

    if args.dataset in ["citeseer","pubmed"]:
        plt.figure(constrained_layout=True, figsize=(10, 6))  #
    else:
        plt.figure(constrained_layout=True, figsize=(8, 6))  #


    # plot figures----------------
    color_code =['#8f8787', '#ffcc5c','#88d8b0','#ff6f69','#92a8d1','#928a97', '#f8b595','#f85f73',]
    if args.include_both:
        sns.scatterplot(x="clean_accuracy", y="certified accuracy", hue="model", style="condition",s=300,
                     data=df_r_all, palette=color_code)  # ,palette= 'Set1'
        plt.title(fr"{name_map[args.dataset]}, ${args.certify_type}$={args.r}")
    else:
        sns.scatterplot(x="clean_accuracy", y="certified accuracy", hue="model", s=300,
                        data=df_r_all, palette=color_code)
        plt.title(fr"{name_map[args.dataset]}, ${args.certify_type}$={args.r}, {args.certify_mode}")
    if args.dataset in ["citeseer","pubmed"]:
        legend =plt.legend(loc='upper left', fancybox=True, bbox_to_anchor=(1.0, 0.9), framealpha=0.5, mode='expend')  #
        for text in legend.get_texts():
            # if text.get_text() == "GCN+SimAug+Conf":
            #     text.set_text('GCN+SimAug\n+Conf')
            if text.get_text() == "GCN+FVAE":
                text.set_text('GCN+FAEAug')
        for legobj in legend.legendHandles:
            legobj.set_sizes([200])
    else:
        legend =plt.legend('', frameon=False)
    if True:
        for index, row in df_r_all.iterrows():
            para_pair=row['parameters'].split(',')[0].split(':')[1]+','+row['parameters'].split(',')[1].split(':')[1]
            # if para_pair!='0.9,0.8':
            plt.text(row["clean_accuracy"], row["certified accuracy"], para_pair, fontdict=font, color='black')
    plt.xlabel(r'clean accuracy')
    plt.ylabel('certified accuracy')
    # plt.arrow(0.68, 0.7, 0.03, 0.3, length_includes_head=False, head_width=0.01, fc='b', ec='b')
    plt.savefig(output_dir + f'clean_certify_curve_{args.certify_type}_{args.r}.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}clean_certify_curve_{args.certify_type}_{args.r}.pdf')
    plt.show()

    return df_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw figures')
    parser.add_argument('-model', type=str, default='GCN', choices=['GCN', 'GAT', 'APPNP'], help='GNN model')
    parser.add_argument('-certify_mode', type=str, default='Vanilla', choices=['Vanilla', 'WithDetectConf'])
    parser.add_argument('-include_both', action='store_true', default=False)
    parser.add_argument('-dataset', type=str, default='citeseer', choices=['cora_ml', 'citeseer', 'pubmed'])
    parser.add_argument('-n_smoothing', type=int, default=10000, help='number of smoothing samples evalute (N)')
    parser.add_argument('-certify_type', type=str, default='r', choices=['r_a', 'r_d'],
                        help='certify delete or add manipulation')

    args = parser.parse_args()
    name_map = {'cora_ml': 'Cora-ML', 'citeseer': 'Citeseer','pubmed':'PubMed'}
    r_dict = {'cora_ml': 5, 'citeseer': 5,'pubmed':5}
    n_dict = {'cora_ml': 10000, 'citeseer': 10000, 'pubmed': 10000}
    MLP_acc = {'cora_ml': 0.691, 'citeseer': 0.660,'pubmed':0.740}
    args.n_smoothing = n_dict[args.dataset]
    args.r = r_dict[args.dataset]

    if args.certify_type=='r_a':
        p_plus_list = [0.1, 0.2, 0.3, 0.4]  #
        p_minus_list = [0.4, 0.5]  #
    else:
        p_plus_list = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4]  #
        p_minus_list = [0.4, 0.5, 0.6, 0.7, 0.8]  #
    combine_list = []
    for p_plus in p_plus_list:
        for p_minus in p_minus_list:
            combine_list.append(f'{p_plus}_0.0_{p_minus}_0.0')
    # combine_list = ['0.6_0.7', '0.7_0.6', '0.7_0.7', '0.7_0.9', '0.8_0.7', '0.9_0.8', '0.9_0.8', '0.9_0.9']
    df_all = clean_certify_figure(args,combine_list)

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
import brokenaxes
from matplotlib.pyplot import MultipleLocator

rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 27
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
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
interval_map={'cora_ml':[0.70,0.80],'citeseer':[0.65,0.75]}

def filter_map(filter):
    filter_mapdict = {'Conf': 'Confidence', 'Entr': '-Entropy','Homo': 'Homophily', 'NSP': 'NSP', 'Prox1': '-Prox1', 'Prox2': '-Prox2', 'JSD':'-JSDivergence','Others': 'Quality'}
    if filter in ['Conf', 'Entr','Homo', 'Prox1', 'Prox2', 'JSD', 'NSP']:
        return filter_mapdict[filter]
    else:
        return filter_mapdict['Others']

# def direction_map(filter):
#     if filter in ['Prox1', 'Prox2', 'JSD']: #Lower is better
#         return ['High','Low']
#     else: #Higher is better
#         return ['Low','High']

def thre_map(args):
    if args.filter == 'Conf':
        return f'Threshold: Confidence>{np.round(1-args.conf_thre,2)}'
    elif args.filter == 'Homo':
        return f'Threshold: Homophily>{1-args.homo_thre}'
    elif args.filter == 'Prox1':
        return f'Threshold: -Prox>{-args.prox_thre}'
    elif args.filter == 'Prox2':
        return f'Threshold: -Prox2>{-args.prox_thre}'
    elif args.filter == 'JSD':
        return f'Threshold: -JSD>{-args.jsd_thre}'
    elif args.filter == 'NSP':
        return f'Threshold: NSP>{np.round(1-args.nsp_thre,2)}'
    elif args.filter == 'Entr':
        return f'Threshold: -Entropy>{-args.etr_thre}'
    else:
        return f'Threshold: Upper {np.round((1-args.quality_thre)*100,2)}%'
#-----------------------

def get_type_df(args,type,mode="filter"):
    if mode=="filter_name":
        out_dir_type = f'./results_{args.dataset}_{args.model}/{type}/{args.pf_plus_adj}_{args.pf_plus_att}_{args.pf_minus_adj}_{args.pf_minus_att}_{args.n_samples_eval}/'
        f = open(f'{out_dir_type}/certify_result_{args.certify_type}.pkl', 'rb')
        df = pickle.load(f)
        f.close()
    else:#"model_name"
        out_dir_type = f'./results_{args.dataset}_{type}/Vanilla/{args.pf_plus_adj}_{args.pf_plus_att}_{args.pf_minus_adj}_{args.pf_minus_att}_{args.n_samples_eval}/'
        f = open(f'{out_dir_type}/certify_result_{args.certify_type}.pkl', 'rb')
        df = pickle.load(f)
        f.close()
        # df["certify_mode"] = f'{type}-Vanilla'
    return df,out_dir_type

def get_combine_df(args,types):
    df_combine = pd.DataFrame()
    for i,type in enumerate(types):
        try:
            if type in ["Vanilla","WithDetectConf","WithDetectEntr", "WithDetectHomo"]:
                df,out_dir_type=get_type_df(args, type, mode="filter_name")
            else:
                df,out_dir_type=get_type_df(args, type, mode="model_name")
            if i==0:
                df_combine = df
            else:
                df_combine = pd.concat([df_combine, df], ignore_index=True)
            if i == len(types)-1:
                f = open(f'{out_dir_type}/smoothing_result.pkl', 'rb')
                _, votes_vanilla = pickle.load(f) #for pA distribution visualization
                f.close()
        except:
            print(f"not found {type}")
            pass
    return df_combine, votes_vanilla


def analyze_result(analysis_data,output_dir,args):
    df = pd.DataFrame(analysis_data)
    df = df.sample(n=30000)
    df["correct_pred"] = df["class_pred"]==df["class_label"]
    #distribution
    if args.filter in ['Entr', 'JSD','Prox1','Prox2']:
        df['quality_score']=-df['anomaly_score']
    else:
        df['quality_score']=1-df['anomaly_score']
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(f"{filter_map(args.filter)} distribution")
    plt.xlabel(f'{filter_map(args.filter)}')
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
               'anomaly predction': [f'Low {filter_map(args.filter)}', f'High {filter_map(args.filter)}']}

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
    plt.title(fr"{name_map[args.dataset]},$(p_+,p_-)$=({args.pf_plus_adj},{args.pf_minus_adj}),N={args.N}")
    plt.savefig(output_dir + f'certifiedaccuracy_{args.certify_type}.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}certifiedaccuracy_{args.certify_type}.pdf')
    plt.show()

def merged_certified_curve(df_combine,output_dir,args):
    # type_map={'GCN-Vanilla':'GCN','GCNJaccard_Aug-Vanilla':'GCN+JacAug','+FilterConf':'GCN+JacAug+Conf'}
    # df_combine=df_combine.replace(type_map)
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{name_map[args.dataset]},$(p_+,p_-)$=({args.pf_plus_adj},{args.pf_minus_adj}),N={args.N}")
    sns.lineplot(x=args.certify_type, y="certified accuracy",hue='certify_mode', palette=['#8f8787', '#ffcc5c','#88d8b0','#ff6f69','#92a8d1','#aa96da'], data=df_combine,linewidth = 3)
    plt.xlabel(f'${args.certify_type}$', fontsize=28)
    plt.legend(loc='lower right', fancybox=True,framealpha=0.5)  #
    plt.savefig(output_dir + f'merged_certaccuracy_{args.certify_type}.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}merged_certaccuracy_{args.certify_type}.pdf')
    plt.show()

def pA_distribution(votes_vanilla,votes,labels, args):
    labels=labels.cpu().numpy()
    pY = [votes[i,l] for i,l in enumerate(labels)]/votes.sum(1)
    pA = np.max(votes,axis=1) / votes.sum(1)
    df = pd.DataFrame({'nodes':range(votes.shape[0]),'p_A':pA,'p_Y':pY})

    pY_vanilla = [votes_vanilla[i, l] for i, l in enumerate(labels)] / votes_vanilla.sum(1)
    pA_vanilla = np.max(votes_vanilla, axis=1) / votes_vanilla.sum(1)
    df_vanilla = pd.DataFrame({'nodes': range(votes_vanilla.shape[0]), 'p_A': pA_vanilla, 'p_Y': pY_vanilla})

    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{name_map[args.dataset]}, $p_A$ distribution")
    sns.histplot(data=df_vanilla, x="p_A", color="dimgrey", label='Vanilla', stat="density", kde=True, bins=30)
    sns.histplot(data=df, x="p_A", color="dodgerblue",label=f'+Filter-{args.filter}', stat="density",kde=True, bins=30)
    plt.xlabel('$p_A$', fontsize=28)
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

def parameter_analysis(args,confs):
    model='GCNSimAug'
    certify_mode='WithDetectConf'
    certify_type='r_d'
    paras = '0.0_0.0_0.8_0.0_10000'
    root_dir = f'./results_{args.dataset}_{model}/{certify_mode}/{paras}'

    output_dir = f'./combined_figures/{args.dataset}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load data----------------certify_mode='Vanilla'
    df_combine = pd.DataFrame()
    for conf in confs:
        try:
            file_dir = f'/certify_result_{certify_type}_{conf}.pkl'
            df1 = pd.DataFrame(load_data(root_dir,file_dir))
            df1['parameters'] = f"$p_+$:{paras.split('_')[0]},$p_-$:{paras.split('_')[2]}"
            df1['model'] = model_map[model]
            df1['condition'] = f'Conf>{round(1-conf,2)}'
            df_combine = pd.concat([df_combine, df1], ignore_index=True)
        except:
            print(f"not found {model},{conf}, vanilla")
            pass

    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{name_map[args.dataset]},$(p_+,p_-)$=(0.0,0.8)")
    sns.lineplot(x=certify_type, y="certified accuracy",hue='condition',palette='Set2', data=df_combine,linewidth = 3)
    plt.xlabel(f'${certify_type}$', fontsize=28)
    plt.legend(loc='lower left', fancybox=True,framealpha=0.5)  #
    plt.savefig(output_dir + f'conf_{certify_type}.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}conf_{certify_type}.pdf')
    plt.show()
def clean_certify_figure(args,combine_list):
    # color_code = ['#740001','#ae0001' ,'#eeba30','#d3a625','#000000','#0e1a40','#222f5b','#5d5d5d','#946b2d']
    # color_code = ["#00aedb","#f47835","#d41243","#8ec127","#5d5d5d","#a200ff","#ffbf00","#222f5b","#946b2d"]
    # color_code = ['#e1f7d5','#ffbdbd',"#c9c9ff","#f1cbff","#ff71ce","#01cdfe","#05ffa1","#b967ff","#fffb96"]
    # color_code = ['#740001','#ae0001' ,'#eeba30','#d3a625',"#5d5d5d","#a200ff","#ffbf00","#222f5b","#946b2d"]
    # color_code = ['#740001','#ae0001' ,'#eeba30','#d3a625','#d11141','#00b159','#00aedb','#f37735','#ffc425']

    root_dir = './results_{dataset}_{model}/{certify_mode}/{paras}'
    file_dir = f'/certify_result_{args.certify_type}.pkl'
    output_dir = f'./combined_figures/{args.dataset}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #load data----------------certify_mode='Vanilla'
    df_combine1 = pd.DataFrame()
    for model in ['GCN', 'GCNJacAug', 'GCNFAEAug','GCNSimAug']:
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
    for model in ['GCN', 'GCNJacAug', 'GCNFAEAug','GCNSimAug']:
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
    # df_r_all = df_r_all.loc[df_r_all['clean_accuracy'] >= MLP_acc[args.dataset]]
    # df_r_all = df_r_all.loc[df_r_all['certified accuracy'] >= 0.05]
    # params = list(df_r_all.loc[df_r_all['certified accuracy'] >= 0.01]['parameters'])
    # df_r_all = df_r_all.loc[df_r_all['parameters'].isin(params)]

    # plot figures----------------
    if args.dataset in ["citeseer"]:#,"pubmed"
        plt.figure(constrained_layout=True, figsize=(10, 6))  #
    else:
        plt.figure(constrained_layout=True, figsize=(8, 6))  #
    unique_models = df_r_all['model'].nunique()
    color_code = ['#8f8787', '#ffcc5c', '#88d8b0', '#ff6f69', '#92a8d1', '#928a97', '#f8b595', '#f85f73'][
                 :unique_models]  #
    if args.certify_type=='r_d':
        df_r_all = df_r_all.loc[df_r_all['clean_accuracy'] >= MLP_acc[args.dataset]]
        params = list(df_r_all.loc[df_r_all['certified accuracy'] >= 0.01]['parameters'])
        df_r_all = df_r_all.loc[df_r_all['parameters'].isin(params)]
        if args.include_both:
            sns.scatterplot(x="clean_accuracy", y="certified accuracy", hue="model", style="condition",s=300,
                         data=df_r_all, palette=color_code)  # ,palette= 'Set1'
            plt.title(fr"{name_map[args.dataset]}, ${args.certify_type}$={args.r}")
        else:
            sns.scatterplot(x="clean_accuracy", y="certified accuracy", hue="model", s=300,
                            data=df_r_all, palette=color_code, ax=bax)
            plt.title(fr"{name_map[args.dataset]}, ${args.certify_type}$={args.r}, {args.certify_mode}")
        if args.dataset in ["citeseer"]:#,"pubmed"
            legend =plt.legend(loc='upper left', fancybox=True, bbox_to_anchor=(1.0, 0.9), framealpha=0.5, mode='expend')  #
            for legobj in legend.legendHandles:
                legobj.set_sizes([200])
        else:
            legend =plt.legend('', frameon=False)
        if True:
            for index, row in df_r_all.iterrows():
                para_pair=row['parameters'].split(',')[0].split(':')[1]+','+row['parameters'].split(',')[1].split(':')[1]
                plt.text(row["clean_accuracy"], row["certified accuracy"], para_pair, fontdict=font, color='black')
        plt.xlabel(r'clean accuracy')
        plt.ylabel('certified accuracy')
        plt.gca().xaxis.set_major_locator(MultipleLocator(0.03))
        plt.savefig(output_dir + f'clean_certify_curve_{args.certify_type}_{args.r}.pdf', dpi=300, bbox_inches='tight')
        print(f'Save result to {output_dir}clean_certify_curve_{args.certify_type}_{args.r}.pdf')
        plt.show()
    else:
        f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, gridspec_kw={'hspace': 0.05, 'wspace': 0.05, 'width_ratios': [1, 2]})
        if args.include_both:
            sns.scatterplot(x="clean_accuracy", y="certified accuracy", hue="model", style="condition",s=300,
                         data=df_r_all, palette=color_code, ax=ax1)
            sns.scatterplot(x="clean_accuracy", y="certified accuracy", hue="model", style="condition", s=300,
                            data=df_r_all, palette=color_code, ax=ax2)
            ax1.set_xlim(0,0.2)
            if args.dataset=="citeseer":
                ax2.set_xlim(0.65,0.76)
            elif args.dataset=="cora_ml":
                ax2.set_xlim(0.68,0.78)
            else:
                ax2.set_xlim(0.76,0.82)
            ax1.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='x', rotation=45)
            ax1.legend_.remove()
            ax2.legend_.remove()
            ax1.set_xlabel('')
            ax2.set_xlabel('')
            ax1.tick_params(top=True, bottom=True, left=False, right=False)
            ax2.tick_params(top=True, bottom=True, left=False, right=False)
            sns.despine(ax=ax1,top=False, bottom=False, left=False, right=True)
            sns.despine(ax=ax2, top=False, bottom=False, left=True, right=False)
            ax1.xaxis.set_major_locator(MultipleLocator(0.08))
            ax2.xaxis.set_major_locator(MultipleLocator(0.04))
            plt.title(fr"{name_map[args.dataset]}, ${args.certify_type}$={args.r}")
            #黑色截断线----------------
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1 - d, 1 + d), (- d, + d), **kwargs)
            ax1.plot((1 - d, 1 + 2*d), (1- d, 1 + d), **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax2.plot((-d, +0.8*d), (-d, +d), **kwargs)
        else:
            sns.scatterplot(x="clean_accuracy", y="certified accuracy", hue="model", s=300,
                            data=df_r_all, palette=color_code, ax=bax)
            plt.title(fr"{name_map[args.dataset]}, ${args.certify_type}$={args.r}, {args.certify_mode}")
        if args.dataset in ["citeseer"]:#,"pubmed"
            legend =plt.legend(loc='upper left', fancybox=True, bbox_to_anchor=(1.15, 0.9), framealpha=0.5, mode='expend')  #
            for legobj in legend.legendHandles:
                legobj.set_sizes([200])
        else:
            legend =plt.legend('', frameon=False)
        if True:
            for index, row in df_r_all[df_r_all['model']!='GCN'].iterrows():
                para_pair=row['parameters'].split(',')[0].split(':')[1]+','+row['parameters'].split(',')[1].split(':')[1]
                plt.text(row["clean_accuracy"], row["certified accuracy"], para_pair, fontdict=font, color='black')
        # plt.xlabel(r'clean accuracy')
        f.text(0.5, -0.05, 'clean accuracy', ha='center', va='center')
        plt.ylabel('certified accuracy')
        plt.savefig(output_dir + f'clean_certify_curve_{args.certify_type}_{args.r}.pdf', dpi=300, bbox_inches='tight')
        print(f'Save result to {output_dir}clean_certify_curve_{args.certify_type}_{args.r}.pdf')
        plt.show()

    return df_all,output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw figures')
    parser.add_argument('-model', type=str, default='GCN', choices=['GCN', 'GAT', 'APPNP'], help='GNN model')
    parser.add_argument('-certify_mode', type=str, default='Vanilla', choices=['Vanilla', 'WithDetectConf'])
    parser.add_argument('-include_both', action='store_true', default=True)
    parser.add_argument('-dataset', type=str, default='cora_ml', choices=['cora_ml', 'citeseer', 'pubmed'])
    parser.add_argument('-n_smoothing', type=int, default=10000, help='number of smoothing samples evalute (N)')
    parser.add_argument('-certify_type', type=str, default='r_a', choices=['r_a', 'r_d'],
                        help='certify delete or add manipulation')

    args = parser.parse_args()
    name_map = {'cora_ml': 'Cora-ML', 'citeseer': 'Citeseer','pubmed':'PubMed'}
    r_dict = {'cora_ml': 5, 'citeseer': 5, 'pubmed':3}
    n_dict = {'cora_ml': 10000, 'citeseer': 10000, 'pubmed': 10000}
    MLP_acc = {'cora_ml': 0.691, 'citeseer': 0.660, 'pubmed':0.740}
    args.n_smoothing = n_dict[args.dataset]
    args.r = r_dict[args.dataset]

    confs=[0.3,0.4,0.5,0.6,0.7]
    parameter_analysis(args,confs)
    if args.certify_type=='r_a':
        p_plus_list = [0.0, 0.01, 0.1, 0.2,  0.4]  #0.1, 0.2, 0.3, 0.4
        p_minus_list = [0.0, 0.2, 0.4, 0.6]  #0.0, 0.2, 0.3, 0.4, 0.5, 0.6
    else:
        p_plus_list = [0.0, 0.01, 0.1]  #0.01, 0.2, 0.3,0.4
        p_minus_list = [0.4, 0.6,0.7, 0.8]  #
    combine_list = []
    for p_plus in p_plus_list:
        for p_minus in p_minus_list:
            combine_list.append(f'{p_plus}_0.0_{p_minus}_0.0')
    # combine_list = ['0.6_0.7', '0.7_0.6', '0.7_0.7', '0.7_0.9', '0.8_0.7', '0.9_0.8', '0.9_0.8', '0.9_0.9']
    df_all,output_dir = clean_certify_figure(args,combine_list)

    table = df_all.pivot_table(index=['parameters','model','condition'], columns=args.certify_type, values='certified accuracy',sort=False)
    print(table)
    table.loc[:, [0, 5, 10, 20, 30]].to_excel(f'{output_dir}/results_table_{args.certify_type}.xlsx')



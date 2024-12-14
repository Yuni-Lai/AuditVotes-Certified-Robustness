import numpy as np
import pandas as pd
from sparse_auditvotes.cert import binary_certificate, joint_binary_certficate
from sparse_auditvotes.data import load_and_standardize, split
from sparse_auditvotes.utils import accuracy_majority, sample_perturbed_mnist,sample_batch_pyg, get_mnist_dataloaders


# def certifiy_type(args):
#     '''certify delete edges or add edegs'''
#     if args.pf_minus_adj>args.pf_plus_adj:
#         return 'r_d'
#     else:
#         return 'r_a'

def certify(pre_votes,votes,labels,idx,args):
    acc_majority = {}
    for split_name in ['train', 'val', 'test']:
        acc_majority[split_name] = accuracy_majority(
            votes=votes, labels=labels, idx=idx[split_name])

    votes_max = votes.max(1)[idx['test']]

    agreement = (votes.argmax(1) == pre_votes.argmax(1)).mean()

    # we are perturbing ONLY the ATTRIBUTES
    if args.pf_plus_adj == 0 and args.pf_minus_adj == 0:
        print('Just ATT')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=args.n_samples_eval, conf_alpha=args.conf_alpha,
            pf_plus=args.pf_plus_att, pf_minus=args.pf_minus_att,type=args.certify_type)
    # we are perturbing ONLY the GRAPH
    elif args.pf_plus_att == 0 and args.pf_minus_att == 0:
        print('Just ADJ')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, conf_alpha=args.conf_alpha,
            pf_plus=args.pf_plus_adj, pf_minus=args.pf_minus_adj,type=args.certify_type)
    else:
        grid_base, grid_lower, grid_upper = joint_binary_certficate(
            votes=votes, pre_votes=pre_votes, conf_alpha=args.conf_alpha,
            pf_plus_adj=args.pf_plus_adj, pf_minus_adj=args.pf_minus_adj,
            pf_plus_att=args.pf_plus_att, pf_minus_att=args.pf_minus_att)

    if args.certify_type=='r_d':
        certi_ratio = (grid_base > 0.5).mean(0)# or grid_lower >= grid_upper (Very Similar result).
        grid_base_acc = (grid_base > 0.5)[idx['test'], :, :][
                        (np.argmax(votes, axis=1) == labels)[idx['test']], :, :]
        certi_acc = np.sum(grid_base_acc, axis=0) / len(idx['test'])
        result_dict = {args.certify_type: range(grid_base.shape[2]), 'certified ratio': certi_ratio[0],
                       'certified accuracy': certi_acc[0]}
    elif args.certify_type=='r_a':
        certi_ratio = (grid_base > 0.5).mean(0)
        grid_base_acc = (grid_base > 0.5)[idx['test'], :, :][
                        (np.argmax(votes, axis=1) == labels)[idx['test']], :, :]
        certi_acc = np.sum(grid_base_acc, axis=0) / len(idx['test'])
        result_dict = {args.certify_type: range(grid_base.shape[1]),
                       'certified ratio': certi_ratio.squeeze(),
                       'certified accuracy': certi_acc.squeeze()}

    df = pd.DataFrame(result_dict)

    mean_max_ra_base = (grid_base > 0.5)[:, :, 0].argmin(1).mean()
    mean_max_rd_base = (grid_base > 0.5)[:, 0, :].argmin(1).mean()
    mean_max_ra_loup = (grid_lower >= grid_upper)[:, :, 0].argmin(1).mean()
    mean_max_rd_loup = (grid_lower >= grid_upper)[:, 0, :].argmin(1).mean()

    results_summary = {
        'acc_majority_train': acc_majority['train'],
        'acc_majority_val': acc_majority['val'],
        'acc_majority_test': acc_majority['test'],
        'mean_max_ra_base': mean_max_ra_base,
        'mean_max_rd_base': mean_max_rd_base,
        'mean_max_ra_loup': mean_max_ra_loup,
        'mean_max_rd_loup': mean_max_rd_loup,
        'agreement': agreement,
    }
    if args.augmenter!='':
        df['certify_mode']=f'{args.model}+{args.augmenter}'
    else:
        df['certify_mode']=f'{args.model}'
    if args.filter !='':
        if len(args.filter.split('_'))>1:
            filter_name=args.filter.split('_')[0]
            df['certify_mode'] = df['certify_mode'] + f'+{filter_name}'
        else:
            df['certify_mode'] = df['certify_mode']+f'+{args.filter}'
    return df, results_summary
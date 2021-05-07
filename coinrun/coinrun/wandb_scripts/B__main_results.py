import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

import time
import glob

from download_wandb_data import load_WandB_csvs

AGENTS = ["ppo_goal_bogdan","ppo",'ppo_diayn',"ppo_curl",'ppo_skrl',"ppg"]
CLEAN_NAMES_AGENTS = ['Ours',"PPO",'PPO+DIAYN',"PPO+CURL",'PPO+Sinkhorn',"DAAC"]

selected_run_ids = {    
                        # 'ppo_goal_bogdan':'(.*jointSKMYOW__2__3__0.300000__200.*)|(.*skrl.*)', 
                        # 'ppo_goal_bogdan':'(.*normUt__2__3__0.500000__200.*)|(.*skrl.*)',
                        'ppo_goal_bogdan':'(.*noShuffleT__3__2__0.100000__200.*)|(.*skrl.*)',
                        # 'ppo_goal_bogdan':'(.*meta.*)|(.*skrl.*)',
                        'ppo_goal':'.*',
                        'ppo':'.*',
                        'ppo_diayn':'.*',
                        'ppg':'.*',
                        'ppo_curl':'.*'}

agent2label = dict(zip(AGENTS,CLEAN_NAMES_AGENTS))

files = glob.glob('../wandb_data/*.csv')
params_to_load = ['agent','n_skills','n_knn','cluster_t','temp','environment']

df = load_WandB_csvs(files,params_to_load,selected_run_ids,AGENTS,agent2label)
df.loc[df['group'].apply(lambda x: 'skrl' in x),'agent'] = 'PPO+Sinkhorn'

print('#######################')
print('Groups:')
print(df['group'].unique())
print('#######################')

# metrics = ['eprew','eprew_eval','silhouette_score']

metrics = ['eprew_eval'] # ,'eprew'

X_LEFT = 0
X_RIGHT = 8e6
REGION = 1e6
EMA = 20

rp = [-1029.86559098,  2344.5778132 , -1033.38786418,  -487.3693808 ,
         298.50245209,   167.25393272]
gp = [  551.32444915, -1098.30287507,   320.71732031,   258.50778539,
         193.11772901,    30.32958789]
bp = [  222.95535971, -1693.48546233,  2455.80348727,  -726.44075478,
         -69.61151887,    67.591787  ]

def clamp(n):
    return min(255, max(0, n))

def gradient(x, rfactors, gfactors, bfactors):
    '''
    Return the r,g,b values along the predefined gradient for
    x in the range [0.0, 1.0].
    '''
    n = len(rfactors)
    r = clamp(int(sum(rfactors[i] * (x**(n-1-i)) for i in range(n))))
    g = clamp(int(sum(gfactors[i] * (x**(n-1-i)) for i in range(n))))
    b = clamp(int(sum(bfactors[i] * (x**(n-1-i)) for i in range(n))))
    return r, g, b


def set_run_attributes(agent):
    
    linewidth = 2
    linestyle = '-'
    do_not_plot = False
    marker = None
    label = agent
    if label == 'Ours':
        col = 'red'
    elif label == 'PPO':
        col = 'limegreen'
        linestyle = '-'
    elif label == 'PPO+DIAYN':
        col = 'fuchsia'
        linestyle = '-'
    elif label == 'DAAC':
        col = 'orange'
        linestyle = '-'
    elif label == 'PPO+CURL':
        col = 'coral'
        linestyle = '-'
    elif label == 'PPO+Sinkhorn':
        col = 'cyan'
        linestyle = '-'
    

    # label = agent2label[label]
    return label, col, linestyle, linewidth, marker


table_eval = r"""
\begin{table}[ht]
\centering
\caption{Average eval returns collected after 25M of training frames, $\pm$ one standard deviation.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l||l|llll}
\toprule
"""
table_train = r"""
\begin{table}[ht]
\centering
\caption{Average train returns collected after 25M of training frames, $\pm$ one standard deviation.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l||l|llll}
\toprule
"""
table_eval += "Env & " + ' & '.join(CLEAN_NAMES_AGENTS) +'\\\\ \n'
reported_scores_eval = np.zeros((16,len(CLEAN_NAMES_AGENTS)))

table_train += "Env & " + ' & '.join(CLEAN_NAMES_AGENTS) +'\\\\ \n'
reported_scores_train = np.zeros((16,len(CLEAN_NAMES_AGENTS)))

for metric in metrics:
    games_list = sorted(df['environment'].unique())
    n_games = len(games_list)    
    nrows = int( np.sqrt(n_games) )
    ncols = n_games // nrows 
    print('Metric: %s, Games: %d, rows: %d, cols: %d' % (metric,n_games,nrows,ncols) )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(12, 12), sharex=False, sharey=False)
    
    row_idx, col_idx = 0,0
    handles_, labels_ = [], []
    for e_idx,env_name in enumerate(games_list):
        if metric == 'eprew_eval':
            row_str_eval = env_name + ' '
        if metric == 'eprew':
            row_str_train = env_name + ' '

        ax = axes[row_idx][col_idx]

        col_idx += 1
        if col_idx >= ncols:
            col_idx = 0
            row_idx += 1
        min_y, max_y = float('inf'), float('-inf')
        
        game_df = df[df['environment']==env_name]
        
        methods_results = {}
        for agent,group_df in game_df.groupby('agent'):
            # agent = group_df['agent'].unique().item()
            print(env_name,agent)
            group_df[metric] = group_df[metric].ewm(EMA).mean()
            
            smallest_t =  group_df.groupby('UID').apply(len).min()
            acc = np.zeros((len(group_df['UID'].unique()),smallest_t))
            for i,seed in enumerate(group_df['UID'].unique()):
                seed_df=group_df[group_df['UID']==seed].iloc[:smallest_t].reset_index()
                x = seed_df['custom_step'].to_numpy()
                acc[i] = seed_df[metric].to_numpy()

            mu = acc.mean(0)
            std = acc.std(0)
            # smooth_df_mu = group_df.groupby('custom_step').apply(np.mean)
            # smooth_df_std = group_df.groupby('custom_step').apply(np.std)

            label, col, linestyle, linewidth, marker = set_run_attributes(agent)
            
            # x = smooth_df_mu.index.to_numpy()
            # mu = smooth_df_mu[metric].to_numpy()
            # std = smooth_df_std[metric].to_numpy()
            ax.plot(x,mu,color=col,label=label,linestyle=linestyle,linewidth=2)
            ax.fill_between(x, mu-std, mu+std, alpha=0.1, color=col)
            max_y = max(max_y,(mu).max())
            min_y = min(min_y,(mu).min())

            left_side = min(x.max()-REGION,X_RIGHT-REGION)
            start_idx = np.min(np.where(np.logical_and(left_side<x,x<X_RIGHT))[0])
            end_idx = np.max(np.where(np.logical_and(left_side<x,x<X_RIGHT))[0])
            # start_idx = len(mu)-10
            best_idx = mu[start_idx:end_idx].argmax()+start_idx
            methods_results[label] = (mu[best_idx],std[best_idx])
            
        a_idx = 0
        for agent in CLEAN_NAMES_AGENTS:
            if metric == 'eprew_eval':
                row_str_eval += '& %.1f$\pm$%.1f ' % methods_results[agent]
            if metric == 'eprew':
                row_str_train += '& %.1f$\pm$%.1f ' % methods_results[agent]
            # add best score to table of scores for normalized absolute score computation
            if metric == 'eprew_eval':
                reported_scores_eval[e_idx,a_idx] = methods_results[agent][0]
            if metric == 'eprew':
                reported_scores_train[e_idx,a_idx] = methods_results[agent][0]
            a_idx += 1

        major_ticks = np.arange(X_LEFT, X_RIGHT+100, 1e6)

        if (row_idx == 3 and e_idx != 11) or e_idx == 15:
            ax.set_xticks(major_ticks)
        else:
            ax.set_xticks([])
        
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e6))
        ax.xaxis.set_major_formatter(ticks_x)
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15) 

        try:
            ax.set_ylim(min_y-(max_y-min_y)*0.1,max_y+(max_y-min_y)*0.1)
        except:
            pass
        ax.set_xlim(X_LEFT,X_RIGHT)
        
        ax.set_title(env_name)
        if (row_idx == 3 and e_idx != 11) or e_idx == 15:
            ax.set_xlabel('Timesteps (M)')
        if col_idx == 1:
            if metric == 'eprew_eval':
                ax.set_ylabel('Average test reward')
            if metric == 'eprew':
                ax.set_ylabel('Average train reward')

        handles, labels = ax.get_legend_handles_labels()

        if metric == 'eprew_eval':
            row_str_eval += '\\\\ \n'
            table_eval += row_str_eval
        if metric == 'eprew':
            row_str_train += '\\\\ \n'
            table_train += row_str_train
        # plt.tight_layout()

    
    plt.legend(handles,labels, loc='upper center', bbox_to_anchor=(-1.5, -0.2),fancybox=False, shadow=False,prop={'size': 15},ncol=len(handles)) # new_handles, new_labels,
    plt.tight_layout()
    
    plt.savefig(os.path.join('..','wandb_plots','B__main_results_'+metric+'.png'),dpi=200)

normed_scores_eval = (reported_scores_eval/np.tile(reported_scores_eval[:,1].reshape(16,1),(1,len(CLEAN_NAMES_AGENTS)))) # standardized by PPO
normed_scores_eval[normed_scores_eval == 0] = np.nan
normed_scores_eval = np.nanmean(normed_scores_eval,0)
table_eval += '\\midrule \n'
table_eval += 'Norm.score & ' + ' & '.join(normed_scores_eval.round(3).astype(str)) + r'\\ \bottomrule'
table_eval += r"""
\end{tabular}%
}
\label{tab:procgen}
\end{table}%
"""

normed_scores_train = (reported_scores_train/np.tile(reported_scores_train[:,1].reshape(16,1),(1,len(CLEAN_NAMES_AGENTS) ) )) # standardized by PPO
normed_scores_train[normed_scores_train == 0] = np.nan
normed_scores_train = np.nanmean(normed_scores_train,0)
table_train += '\\midrule \n'
table_train += 'Norm.score & ' + ' & '.join(normed_scores_train.round(3).astype(str)) + r'\\ \bottomrule'
table_train += r"""
\end{tabular}%
}
\label{tab:procgen}
\end{table}%
"""
print(table_eval)
# print(table_train)
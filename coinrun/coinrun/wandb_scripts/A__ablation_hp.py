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

# ablation = 'cluster_sim'
# ablation = 'cluster_t'
# ablation = 'n_clusters'
# ablation = 'temp'
ablation = "loss"

AGENTS = ["ppo_goal_bogdan"]
CLEAN_NAMES_AGENTS = ['Ours']

if ablation == "temp":
    run_regex = '(.*jointSKMYOW__2__3__.*)'
elif ablation == "n_clusters":
    run_regex = '.*normUt__2__3__0.300000__(100|300).*'
elif ablation == "cluster_t":
    run_regex = '(.*ShuffleT__3__2__0.100000__200.*)|(.*ShuffleT__5__3__0.300000__200.*)'
elif ablation == 'cluster_sim':
    run_regex = '(.*__trajectoryClusters.*)'
elif ablation == 'loss':
    run_regex = '(.*jointSKMYOW__2__3__0.200000__200.*)'
else:
    print('Not an existing ablation')
    exit()

selected_run_ids = {
                        # 'ppo_goal_bogdan':'(.*smarterMYOW__5__3__.*)',
                        'ppo_goal_bogdan':run_regex
                    }

agent2label = dict(zip(AGENTS,CLEAN_NAMES_AGENTS))

files = glob.glob('../wandb_data/*.csv')
params_to_load = ['agent','n_skills','n_knn','cluster_t','temp','environment']

df = load_WandB_csvs(files,params_to_load,selected_run_ids,AGENTS,agent2label)

print('#######################')
print('Groups:')
print(df['group'].unique())
print('#######################')

# metrics = ['eprew','eprew_eval','silhouette_score']
if ablation == 'cluster_sim':
    metrics = ['cluster_similarity']
elif ablation == 'loss':
    metrics = ["myow_loss","cluster_loss"]
else:
    metrics = ['silhouette_score','eprew_eval','cluster_similarity']
clean_metric = {'eprew':'Training reward',
                'eprew_eval':'Evaluation reward',
                'silhouette_score': 'Silhouette score'}

X_LEFT = 0
X_RIGHT = 8e6
# X_RIGHT = 2000
EMA = 10

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


def set_run_attributes(sub_df):
    """
    Ablation 1: temperature vs silhouette
    """
    if ablation == "temp":
        T = float(sub_df['temp'].unique()[-1])
        label = 'T=%.2f' %(T)
        if T == 0.3:
            col = 'tab:red'
        elif T == 0.1:
            col = 'tab:orange'
        elif T == 0.01:
            col = 'tab:olive'
        elif T == 0.2:
            col = 'tab:blue'
    """
    Ablation 2: number of clusters
    """
    if ablation == "n_clusters":
        n_clusters = float(sub_df['n_skills'].unique()[-1])
        label = 'Number of clusters=%d' %(n_clusters)
        if n_clusters == 200:
            col = 'tab:red'
        elif n_clusters == 100:
            col = 'tab:orange'
        elif n_clusters == 400:
            col = 'tab:olive'
        elif n_clusters == 300:
            col = 'tab:blue'
    if ablation == 'cluster_t':
        cluster_t = float(sub_df['cluster_t'].unique()[-1])
        label = 'Cluster timesteps=%d' %(cluster_t)
        if cluster_t == 2:
            col = 'tab:red'
        elif cluster_t == 3:
            col = 'tab:orange'
        elif cluster_t == 10:
            col = 'tab:olive'
        elif cluster_t == 5:
            col = 'tab:blue'
    if ablation == 'cluster_sim':
        col = 'tab:red'
        label = 'Cluster similarity'
    if ablation == 'loss':
        col = 'tab:red'
        label = 'Loss'
    return label, col, '-'

for metric in metrics:
    games_list = sorted(df['environment'].unique())
    n_games = len(games_list)    
    # n_games = 16
    nrows = int( np.sqrt(n_games) )
    ncols = n_games // nrows 
    print('Metric: %s, Games: %d, rows: %d, cols: %d' % (metric,n_games,nrows,ncols) )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(16, 12), sharex=False, sharey=False)
    
    row_idx, col_idx = 0,0
    handles_, labels_ = [], []
    for e_idx,env_name in enumerate(games_list):
        if metric == 'eprew_eval':
            row_str_eval = env_name + ' '
        if metric == 'eprew':
            row_str_train = env_name + ' '
        if n_games == 1:
            ax = axes
        else:
            ax = axes[row_idx][col_idx]

        col_idx += 1
        if col_idx >= ncols:
            col_idx = 0
            row_idx += 1
        min_y, max_y = float('inf'), float('-inf')
        
        game_df = df[df['environment']==env_name]
        # print(game_df.columns)
        for group_name,group_df in game_df.groupby('group'):
            group_df[metric] = group_df[metric].ewm(EMA).mean()
            # group_df[metric] = group_df[metric].ewm(20).mean()
            # smooth_df_mu = group_df.groupby('custom_step').apply(np.mean)
            # smooth_df_std = group_df.groupby('custom_step').apply(np.std)

            HP_LABEL, col, linestyle = set_run_attributes(group_df)
            
            # x = smooth_df_mu.index.to_numpy()
            # mu = smooth_df_mu[metric].to_numpy()
            # std = smooth_df_std[metric].to_numpy()

            print(env_name,group_name)
            
            
            smallest_t =  group_df.groupby('UID').apply(len).min()
            acc = np.zeros((len(group_df['UID'].unique()),smallest_t))
            for i,seed in enumerate(group_df['UID'].unique()):
                seed_df=group_df[group_df['UID']==seed].iloc[:smallest_t].reset_index()
                x = seed_df['custom_step'].to_numpy()
                acc[i] = seed_df[metric].to_numpy()

            mu = acc.mean(0)
            std = acc.std(0)

            start_idx = np.min(np.where(np.logical_and(X_LEFT<x,x<X_RIGHT))[0])
            end_idx = np.max(np.where(np.logical_and(X_LEFT<x,x<X_RIGHT))[0])

            x = x[start_idx:end_idx]
            mu = mu[start_idx:end_idx]

            x = np.linspace(X_LEFT, X_RIGHT,len(x))
            
            ax.plot(x,mu,color=col,label=HP_LABEL,linestyle=linestyle,linewidth=2)
            # ax.fill_between(x, mu-std, mu+std, alpha=0.1, color=np.array(col)/255.)
            max_y = max(max_y,(mu).max())
            min_y = min(min_y,(mu).min())

        ticks_freq = X_RIGHT // 8

        major_ticks = np.arange(X_LEFT, X_RIGHT, ticks_freq)

        if ((row_idx == 3 and e_idx != 11) or e_idx == 15) or n_games == 1:
            ax.set_xticks(major_ticks)
        else:
            ax.set_xticks([])
        
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/ticks_freq))
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
        
        ax.set_title(env_name,fontweight="bold",fontsize=20)

        if ( (row_idx == 3 and e_idx != 11) or e_idx == 15 ) or n_games == 1:
            ax.set_xlabel('Timesteps (M)',fontsize=17)
        if col_idx == 1 or n_games == 1:
            if metric == 'silhouette_score':
                ax.set_ylabel('Silhouette score',fontsize=17)
            if metric == 'eprew_eval':
                ax.set_ylabel('Average test reward',fontsize=17)

        # ax.set_xlabel('Timesteps (M)',fontsize=17)
        # ax.set_ylabel(clean_metric[metric],fontsize=17)

        handles, labels = ax.get_legend_handles_labels()


    plt.tight_layout()
    # plt.legend(handles,labels, loc='lower left', bbox_to_anchor=(0.6,0.1),prop={'size': 15}) # new_handles, new_labels,

    # plt.legend(handles,labels, loc='upper center', bbox_to_anchor=(-1.5, -0.3),fancybox=False, shadow=False,prop={'size': 15},ncol=len(handles)) # new_handles, new_labels,
    # plt.tight_layout()
    
    plt.savefig(os.path.join('..','wandb_plots','A__ablation_' + ablation + '_'+metric+'.png'),dpi=200)

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

AGENTS = ["ppo_goal_bogdan"]
CLEAN_NAMES_AGENTS = ['Ours']

selected_run_ids = {'ppo_goal_bogdan':'(.*smarterMYOW__5__3__.*)',
                        'ppo':'.*',
                        'ppo_diayn':'.*',
                        'ppg':'.*',
                        'ppo_curl':'.*'}

agent2label = dict(zip(AGENTS,CLEAN_NAMES_AGENTS))

files = glob.glob('../wandb_data/*.csv')
params_to_load = ['agent','n_skills','n_knn','cluster_t','temp','environment']

df = load_WandB_csvs(files,params_to_load,selected_run_ids,AGENTS,agent2label)

print('#######################')
print('Groups:')
print(df['group'].unique())
print('#######################')

metrics = ['eprew','eprew_eval','silhouette_score']

X_LEFT = 0
X_RIGHT = 8e6

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
    T = float(sub_df['temp'].unique()[-1])
    label = 'T=%.1f' %(T)
    col = gradient(T/5, rp, gp, bp)
    return label, col, '-'

for metric in metrics:
    games_list = sorted(df['environment'].unique())
    n_games = len(games_list)    
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

        ax = axes[row_idx][col_idx]

        col_idx += 1
        if col_idx >= ncols:
            col_idx = 0
            row_idx += 1
        min_y, max_y = float('inf'), float('-inf')
        
        game_df = df[df['environment']==env_name]
        
        for group_name,group_df in game_df.groupby('group'):

            group_df[metric] = group_df[metric].ewm(30).mean()
            smooth_df_mu = group_df.groupby('custom_step').apply(np.mean)
            smooth_df_std = group_df.groupby('custom_step').apply(np.std)

            HP_LABEL, col, linestyle = set_run_attributes(smooth_df_mu)
            
            x = smooth_df_mu.index.to_numpy()
            mu = smooth_df_mu[metric].to_numpy()
            std = smooth_df_std[metric].to_numpy()
            ax.plot(x,mu,color=np.array(col)/255.,label=HP_LABEL,linestyle=linestyle,linewidth=2)
            ax.fill_between(x, mu-std, mu+std, alpha=0.1, color=np.array(col)/255.)
            max_y = max(max_y,(mu).max())
            min_y = min(min_y,(mu).min())


        major_ticks = np.arange(X_LEFT, X_RIGHT, 1e6)

        ax.set_xticks(major_ticks)
        
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
        ax.set_xlabel('Timesteps (M)')
        ax.set_ylabel('Average metric')

        handles, labels = ax.get_legend_handles_labels()


    plt.tight_layout()
    plt.legend(handles,labels, loc='lower left', bbox_to_anchor=(0.5,0.5),prop={'size': 15}) # new_handles, new_labels,
    
    plt.savefig(os.path.join('..','wandb_plots','A__ablation_hp_'+metric+'.png'),dpi=200)

# for name, groups in df.groupby(['group','custom_step']):
#     import ipdb;ipdb.set_trace()


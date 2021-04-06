import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
os.environ["WANDB_API_KEY"] = "02e3820b69de1b1fcc645edcfc3dd5c5079839a1"
os.environ["WANDB_API_KEY"] = "02e3820b69de1b1fcc645edcfc3dd5c5079839a1"
import wandb
api = wandb.Api()

def set_curve_attributes(params):
        linewidth = 2
        linestyle = '-'
        do_not_plot = False
        marker = None
        label = params['agent']['value']
        if label == 'ppo_goal':
            col = 'red'
            label = 'Ours'
        elif label == 'ppo':
            col = 'limegreen'
            linestyle = '--'
            label = 'PPO'
        elif label == 'ppo_diayn':
            col = 'fuchsia'
            linestyle = '--'
            label = 'PPO+DIAYN'
        elif label == 'ppg':
            col = 'orange'
            linestyle = '--'
            label = 'DAAC'
        
        return label, col, linestyle, linewidth, marker, do_not_plot

runs = api.runs("ssl_rl/procgen_generalization")
df = []
for i,run in enumerate(runs): 
    params = json.loads(run.json_config)
    if params['agent']['value'] not in ("ppo_goal","ppo","ppo_diayn"):
        continue
    label, col, linestyle, linewidth, marker, do_not_plot = set_curve_attributes(params)
    run_df = run.history()
    columns = run_df.columns
    run_df.columns = [x.split('/')[-1] for x in columns]

    run_df['agent'] = label
    run_df['env'] = params['environment']['value']
    run_df['col'] = col
    run_df['linestyle'] = linestyle
    run_df['UID'] = np.random.randint(1000000)
    df.append(run_df)
df=pd.concat(df)

xlim = (0,25e6,5e6)

metrics = ['eprew','eprew_eval']
for metric in metrics:
    games_list = sorted(df['env'].unique())
    n_games = len(games_list)    
    nrows = int( np.sqrt(n_games) )
    ncols = n_games // nrows 
    print('Metric: %s, Games: %d, rows: %d, cols: %d' % (metric,n_games,nrows,ncols) )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(20, 12), sharex=False, sharey=False)
    
    row_idx, col_idx = 0,0
    handles_, labels_ = [], []
    for env_name in games_list:
        # idx = ((df['env']==env_name))
        # df_game = df[idx]

        ax = axes[row_idx][col_idx]

        col_idx += 1
        if col_idx >= ncols:
            col_idx = 0
            row_idx += 1
        min_y, max_y = float('inf'), float('-inf')
        for agent in sorted(df['agent'].unique()):
            idx2 = ( (df['agent']==agent).to_numpy() * (df['env']==env_name).to_numpy() )
            df_agent = df[idx2]
            T = df_agent.groupby('UID').apply(len).min()
            seeds = df_agent['UID'].unique()
            n_seeds = len(seeds)

            x = np.zeros((n_seeds,T))
            mu = np.zeros((n_seeds,T))
            
            for i,seed in enumerate(seeds):
                df_seed = df_agent[df_agent['UID']==seed]
                x[i] = df_seed['custom_step'].to_numpy()[:T]
                mu[i] = df_seed[metric].ewm(30).mean().to_numpy()[:T]
            
            x = x.mean(0)
            sigma = mu.std(0)
            mu = mu.mean(0)
            col = df_seed['col'].unique().item()
            linestyle = df_seed['linestyle'].unique().item()
            ax.plot(x,mu,color=col,label=agent,linestyle=linestyle,linewidth=2)
            ax.fill_between(x, mu-sigma, mu+sigma, alpha=0.1, color=col)
            max_y = max(max_y,(mu).max())
            min_y = min(min_y,(mu).min())

        major_ticks = np.arange(xlim[0], xlim[1], xlim[2])
        #minor_ticks = np.arange(xlim[0], xlim[1], xlim[3])

        ax.set_xticks(major_ticks)
        #ax.set_xticks(minor_ticks, minor=True)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e6))
        ax.xaxis.set_major_formatter(ticks_x)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15) 

        try:
            ax.set_ylim(min_y-(max_y-min_y)*0.2,max_y+(max_y-min_y)*0.2)
        except:
            pass
        ax.set_xlim(xlim[0],xlim[1])
        
        ax.set_title(env_name)
        ax.set_xlabel('Timesteps (M)')
        ax.set_ylabel('Average metric')

        handles, labels = ax.get_legend_handles_labels()

    plt.tight_layout()
    plt.legend(handles,labels, loc='lower left', bbox_to_anchor=(0.5,0.5)) # new_handles, new_labels,
    
    plt.savefig(os.path.join('wandb_plots',metric+'.png'),dpi=200)
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
import time
import glob

AGENTS = ["ppo_goal_bogdan","ppo","ppo_diayn","ppg","ppo_curl"]
CLEAN_NAMES_AGENTS = ['Ours','PPO','PPO+DIAYN','PPG','PPO+CURL']

agent2label = dict(zip(AGENTS,CLEAN_NAMES_AGENTS))

def set_curve_attributes(agent):
        linewidth = 2
        linestyle = '-'
        do_not_plot = False
        marker = None
        label = agent
        if label == 'ppo_goal':
            col = 'red'
        elif label == 'ppo':
            col = 'limegreen'
            linestyle = '--'
        elif label == 'ppo_diayn':
            col = 'fuchsia'
            linestyle = '--'
        elif label == 'ppg':
            col = 'orange'
            linestyle = '--'
        elif label == 'ppg_ssl':
            col = 'coral'
            linestyle = '-'
        
        label = agent2label[label]
        
        return label, col, linestyle, linewidth, marker, do_not_plot

files = glob.glob('wandb_data/*')

df = []
for i,fh in enumerate(files): 
    run_df = pd.read_csv(fh,index_col='custom_step').drop(['_step','Unnamed: 0'],1)
    run_id = re.findall('(.*__[0-9]+__[0-9]+)',fh)[0].split('/')[1]
    agent = '_'.join(fh.split(run_id)[1][1:].split('_')[:-1])

    
    if agent not in AGENTS:
        continue
    if agent == 'ppo_goal' and run_id != 'ppo_split_sinkhornMYOW':
        continue
    label, col, linestyle, linewidth, marker, do_not_plot = set_curve_attributes(agent)

    run_df['agent'] = label
    run_df['col'] = col
    run_df['linestyle'] = linestyle
    run_df['UID'] = np.random.randint(1000000)
    df.append(run_df)
df=pd.concat(df)
import ipdb;ipdb.set_trace()

xlim = (0,25e6,5e6)

table_eval = r"""
\begin{table}[ht]
\centering
\caption{Average eval returns collected after 25M of training frames, $\pm$ one standard deviation.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l||l|ll}
\toprule
"""
table_train = r"""
\begin{table}[ht]
\centering
\caption{Average train returns collected after 25M of training frames, $\pm$ one standard deviation.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l||l|ll}
\toprule
"""
table_eval += "Env & " + ' & '.join(CLEAN_NAMES_AGENTS) +'\\\\ \n'
reported_scores_eval = np.zeros((16,len(CLEAN_NAMES_AGENTS)))

table_train += "Env & " + ' & '.join(CLEAN_NAMES_AGENTS) +'\\\\ \n'
reported_scores_train = np.zeros((16,len(CLEAN_NAMES_AGENTS)))

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
    for e_idx,env_name in enumerate(games_list):
        if metric == 'eprew_eval':
            row_str_eval = env_name + ' '
        if metric == 'eprew':
            row_str_train = env_name + ' '
        # idx = ((df['env']==env_name))
        # df_game = df[idx]

        ax = axes[row_idx][col_idx]

        col_idx += 1
        if col_idx >= ncols:
            col_idx = 0
            row_idx += 1
        min_y, max_y = float('inf'), float('-inf')
        for a_idx,agent in enumerate(CLEAN_NAMES_AGENTS):
            idx2 = ( (df['agent']==agent).to_numpy() * (df['env']==env_name).to_numpy() )
            df_agent = df[idx2]
            if len(df_agent) == 0:
                continue
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

            # find largest reward in curve in interval [X_LEFT,X_RIGHT]
            X_LEFT = 0
            X_RIGHT = 8e6
            start_idx = np.max(np.where(x<8e6)[0])
            # start_idx = len(mu)-10
            best_idx = mu[:start_idx].argmax()#+start_idx

            if metric == 'eprew_eval':
                row_str_eval += '& %.1f$\pm$%.1f ' % (mu[best_idx],sigma[best_idx])
            if metric == 'eprew':
                row_str_train += '& %.1f$\pm$%.1f ' % (mu[best_idx],sigma[best_idx])
            # add best score to table of scores for normalized absolute score computation
            if metric == 'eprew_eval':
                reported_scores_eval[e_idx,a_idx] = mu[best_idx]
            if metric == 'eprew':
                reported_scores_train[e_idx,a_idx] = mu[best_idx]

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

        if metric == 'eprew_eval':
            row_str_eval += '\\\\ \n'
            table_eval += row_str_eval
        if metric == 'eprew':
            row_str_train += '\\\\ \n'
            table_train += row_str_train


    plt.tight_layout()
    plt.legend(handles,labels, loc='lower left', bbox_to_anchor=(0.5,0.5)) # new_handles, new_labels,
    
    plt.savefig(os.path.join('wandb_plots',metric+'.png'),dpi=200)

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
import ipdb;ipdb.set_trace()

import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
os.environ["WANDB_API_KEY"] = "02e3820b69de1b1fcc645edcfc3dd5c5079839a1"
import wandb
import time
import re

def set_curve_attributes(agent,agent2label):
    linewidth = 2
    linestyle = '-'
    do_not_plot = False
    marker = None
    label = agent
    if label == 'ppo_goal' or label =='ppo_goal_bogdan':
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
    elif label == 'ppo_curl':
        col = 'coral'
        linestyle = '-'
    

    label = agent2label[label]
    
    return label, col, linestyle, linewidth, marker, do_not_plot



if __name__ == "__main__":
     
    LOAD_CSV = True


    if not LOAD_CSV:
        api = wandb.Api()

        AGENTS = ["split",'joint']

        model='spr'

        URL = "ssl_rl/procgen_generalization_driml" if model == 'driml' else "bmazoure/procgen_generalization_spr"

        METRICS = ['ep_rew'] if model == 'driml' else ['EvalReturn']

        CLEAN_NAMES_AGENTS = ['Split updates','Joint updates']

        agent2label = dict(zip(AGENTS,CLEAN_NAMES_AGENTS))

        X_RIGHT = 8e6
        REGION = 8e6
        X_LEFT = 0

        runs = api.runs(URL)
        df = []
        
        for i,run in enumerate(runs): 
        
            params = json.loads(run.json_config)
            
            
            env_name = run.name.split('-')[1] if model == 'driml' else run.name.split('__')[0]
            split = 'phaseSplit' in run.name or '__1__' in run.name
            # env_name = params['environment']['value']
            RUN_METRICS = [env_name+'/'+metric for metric in METRICS]
            run_df = run.history(keys=RUN_METRICS,samples=int(1e6))
            if len(run_df) == 0:
                run_df = run.history(keys=RUN_METRICS[:-1],samples=int(1e6))
                
            columns = run_df.columns
            run_df.columns = [x.split('/')[-1] for x in columns]

            run_df['agent'] = 'split' if split else 'joint'
            run_df['env_name'] = env_name
            run_df['UID'] = np.random.randint(100000)


            # run_df.to_csv(os.path.join('..','wandb_data',filename+'.csv'))

            # with open(os.path.join('..','wandb_data',filename+'.json'), 'w') as f:
            #     json.dump(params, f)
            print('Loaded ',i,run.name,'with %d rows'%len(run_df))

            df.append(run_df) 
            time.sleep(1)
            
        df = pd.concat(df)
        reported_scores_eval = np.zeros((16,len(CLEAN_NAMES_AGENTS)))

        for metric in METRICS:
            games_list = sorted(df['env_name'].unique())
            n_games = len(games_list)    
            nrows = int( np.sqrt(n_games) )
            ncols = n_games // nrows 
            print('Metric: %s, Games: %d, rows: %d, cols: %d' % (metric,n_games,nrows,ncols) )

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(12, 12), sharex=False, sharey=False)
            
            row_idx, col_idx = 0,0
            handles_, labels_ = [], []
            for e_idx,env_name in enumerate(games_list):

                ax = axes[row_idx][col_idx]

                col_idx += 1
                if col_idx >= ncols:
                    col_idx = 0
                    row_idx += 1
                min_y, max_y = float('inf'), float('-inf')
                
                game_df = df[df['env_name']==env_name]
                
                methods_results = {}
                for agent,group_df in game_df.groupby('agent'):
                    # agent = group_df['agent'].unique().item()
                    print(env_name,agent)
                    group_df[metric] = group_df[metric].ewm(20).mean()
                    
                    # smallest_t =  group_df.groupby('UID').apply(len).min()
                    # acc = np.zeros((len(group_df['UID'].unique()),smallest_t))
                    # for i,seed in enumerate(group_df['UID'].unique()):
                    #     seed_df=group_df[group_df['UID']==seed].iloc[:smallest_t].reset_index()
                    #     x = seed_df['custom_step'].to_numpy()
                    #     acc[i] = seed_df[metric].to_numpy()

                    mu = group_df.groupby('_step').apply(np.mean)[metric].to_numpy()
                    std = smooth_df_std = group_df.groupby('_step').apply(np.std)[metric].to_numpy()

                    # mu = acc.mean(0)
                    # std = acc.std(0)
                    # smooth_df_mu = group_df.groupby('custom_step').apply(np.mean)
                    # smooth_df_std = group_df.groupby('custom_step').apply(np.std)

                    # label, col, linestyle, linewidth, marker = set_run_attributes(agent)
                    label = agent2label[agent]
                    
                    # x = smooth_df_mu.index.to_numpy()
                    # mu = smooth_df_mu[metric].to_numpy()
                    # std = smooth_df_std[metric].to_numpy()
                    # ax.plot(x,mu,color=col,label=label,linestyle=linestyle,linewidth=2)
                    # ax.fill_between(x, mu-std, mu+std, alpha=0.1, color=col)
                    max_y = max(max_y,(mu).max())
                    min_y = min(min_y,(mu).min())

                    # left_side = min(x.max()-REGION,X_RIGHT-REGION)
                    # start_idx = np.min(np.where(np.logical_and(left_side<x,x<X_RIGHT))[0])
                    # end_idx = np.max(np.where(np.logical_and(left_side<x,x<X_RIGHT))[0])
                    # start_idx = len(mu)-10
                    # best_idx = mu.argmax()#+start_idx
                    best_idx = -1
                    methods_results[label] = mu[best_idx] #(mu[best_idx],std[best_idx])
                    
                a_idx = 0
                for agent in CLEAN_NAMES_AGENTS:
                    reported_scores_eval[e_idx,a_idx] = methods_results[agent]
                    a_idx += 1

        
        clean_df=pd.DataFrame(reported_scores_eval)
        clean_df.columns=CLEAN_NAMES_AGENTS
        clean_df.to_csv(model+'_Procgen_scores.csv',index=None)
    else:
        table = []
        models = ['Env','driml','spr']
        for model in models:
            if model == 'Env':
                table.append("bigfish bossfight caveflyer chaser climber coinrun dodgeball fruitbot heist jumper leaper maze miner ninja plunder starpilot".split(' '))
            else:
                clean_df = pd.read_csv(model+'_Procgen_scores.csv')

                normed_scores = ((clean_df['Split updates']-clean_df['Joint updates'])/clean_df['Joint updates']).to_numpy().round(2)
                table.append(normed_scores)
        
        table = pd.DataFrame(table).transpose()
        table.columns = models
    print(table.to_latex(index=False))
    import ipdb;ipdb.set_trace()
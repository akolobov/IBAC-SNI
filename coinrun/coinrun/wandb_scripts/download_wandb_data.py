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
    elif label == 'ppo_skrl':
        col = 'cyan'
        linestyle = '-'
    elif label == 'ppo_bisimulation':
        col = 'grey'
        linestyle = '-'
    

    label = agent2label[label]
    
    return label, col, linestyle, linewidth, marker, do_not_plot


def load_WandB_csvs(files,params_to_load,selected_run_ids,AGENTS,agent2label):
    df = []
    for i,fh in enumerate(files): 
        params_selected = {}
        with open(fh[:-4]+'.json','r') as json_fh:
            params = json.load(json_fh)
            for p in params_to_load:
                if p not in params:
                    params[p] = {}
                    params[p]['value'] = 1
                params_selected[p] = params[p]['value']

        run_id_fn, agent_fn = fh.split('/')[2].split('___')
        run_id = params['run_id']['value']
        agent = params['agent']['value']
        agent_fn = agent.split('.')[0]
        group = '__'.join(run_id_fn.split('__')[:-1])

        if agent not in AGENTS:
            continue
        
        if not re.findall(selected_run_ids[agent],group):
            continue

        try:
            run_df = pd.read_csv(fh,index_col='custom_step').drop(['Unnamed: 0'],1)
            # run_df['custom_step'] = run_df['_step']
            run_df = run_df.drop(['_step'],1
            )
        except Exception as e:
            print('Skipping malformated CSV')
            print(e)
            continue
        
        for k,v in params_selected.items():
            run_df[k] = v

        label, col, linestyle, linewidth, marker, do_not_plot = set_curve_attributes(agent,agent2label)

        run_df['agent'] = label
        run_df['col'] = col
        run_df['linestyle'] = linestyle
        run_df['UID'] = np.random.randint(1000000)
        run_df['group'] = group
                
        df.append(run_df)
    df=pd.concat(df)
    return df

if __name__ == "__main__":

    api = wandb.Api()

    AGENTS = ["ppo_bisimulation","ppo_goal_bogdan",'ppo_goal',"ppo","ppo_diayn","ppg",'ppo_curl']

    # METRICS = ['custom_step','eprew','eprew_eval']
    # METRICS = ['custom_step','eprew','eprew_eval','silhouette_score']
    # METRICS = ['cluster_similarity']
    METRICS = ['custom_step','myow_loss','cluster_loss']

    runs = api.runs("ssl_rl/procgen_generalization")
    # runs = api.runs("ssl_rl/ising_generalization")
    df = []
    for i,run in enumerate(runs): 
    
        params = json.loads(run.json_config)
        agent = params['agent']['value']
        if agent not in AGENTS:
            continue
        
        env_name = params['environment']['value']
        RUN_METRICS = [env_name+'/'+metric for metric in METRICS]
        run_df = run.history(keys=RUN_METRICS,samples=int(1e6))
        if len(run_df) == 0:
            run_df = run.history(keys=RUN_METRICS,samples=int(1e6))
            
        columns = run_df.columns
        run_df.columns = [x.split('/')[-1] for x in columns]

        filename = '%s___%s' % (run.name,agent)

        run_df.to_csv(os.path.join('..','wandb_data',filename+'.csv'))

        with open(os.path.join('..','wandb_data',filename+'.json'), 'w') as f:
            json.dump(params, f)
        print('Loaded ',i,run.name,'with %d rows'%len(run_df))
        time.sleep(1)
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
api = wandb.Api()

AGENTS = ["ppo_goal","ppo","ppo_diayn","ppg","ppo_goal_bogdan","ppo_goal_bogdan"]
CLEAN_NAMES_AGENTS = ['Ours','PPO','PPO+DIAYN','PPG','Ours (PPG)']

agent2label = dict(zip(AGENTS,CLEAN_NAMES_AGENTS))

def set_curve_attributes(params):
        linewidth = 2
        linestyle = '-'
        do_not_plot = False
        marker = None
        label = params['agent']['value']
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

METRICS = ['custom_step','eprew','eprew_eval','silhouette_score']

runs = api.runs("ssl_rl/procgen_generalization")
df = []
for i,run in enumerate(runs): 
    print(run.name)
    params = json.loads(run.json_config)
    agent = params['agent']['value']
    if agent not in AGENTS:
        continue
    # if params['agent']['value'] == 'ppo_goal' and params['run_id']['value'] != 'ppo_split_sinkhornMYOW':
    #     continue
    # label, col, linestyle, linewidth, marker, do_not_plot = set_curve_attributes(params)
    env_name = params['environment']['value']
    RUN_METRICS = [env_name+'/'+metric for metric in METRICS]
    run_df = run.history(keys=RUN_METRICS,samples=int(1e6))
    
    columns = run_df.columns
    run_df.columns = [x.split('/')[-1] for x in columns]

    run_df['env'] = env_name

    filename = '%s_%s_%s.csv' % (run.name,agent,env_name)

    run_df.to_csv(os.path.join('wandb_data',filename))
    # if params['agent']['value'] == 'ppo_goal':
    #     import ipdb;ipdb.set_trace()

    # run_df['agent'] = label
    
    # run_df['col'] = col
    # run_df['linestyle'] = linestyle
    # run_df['UID'] = np.random.randint(1000000)
    df.append(run_df)
    time.sleep(1)
df=pd.concat(df)
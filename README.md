# Introduction 

This codebase is for a custom research project, built off of the IBAC-SNI repo


## Plotting

To plot the results, modify the `plots.py` file by changing the `path`, as well as the `experiments`
dictionary to specify which subfolders in `path` you would like to plot.

# Coinrun

Please follow the installation instructions taken from the original repo to install the requirements: 
```
# Linux
apt-get install mpich build-essential qt5-default pkg-config
# Mac
brew install qt open-mpi pkg-config

cd coinrun
pip install tensorflow==1.15.0  # or tensorflow-gpu
pip install -r requirements.txt
pip install -e .
```

Also, in `coinrun/coinrun/config.py` set the `self.WORKDIR` and `self.TB_DIR` variables.

## Reproducing Results Procgen
A sample commnad is here. This repo can use mpi as shown later, but here is an example without it
```
python3 -m coinrun.train_agent --env starpilot --run-id baseline --num-levels 0 --short
```


Depracted: 
To reproduce the results, run on a NC24 with 4 GPUs (3 will be used for training, one for testing):
```
export env=<env>
mpiexec -n 1 python3 -m coinrun.train_agent --env starpilot --run-id baseline --num-levels 0 --short 
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id baseline --num-levels 200 --test --short --l2 0.0001 -uda 1
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id ibac-sni-lambda0.5 --num-levels 200 --test --short --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id ibac-sni-lambda1.0 --num-levels 200 --test --l2 0.0001 -uda 1 --beta-l2a 0.0001 --short
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id ibac --num-levels 200 --test --short --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id dropout0.2-sni-lambda0.5 --num-levels 200 --test --short --l2 0.0001 -uda 1 --dropout 0.2 --sni2
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id dropout0.2-sni-lambda1.0 --num-levels 200 --test --short --l2 0.0001 -uda 1 --dropout 0.2 --openai
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id dropout0.2 --num-levels 200 --test --short --l2 0.0001 -uda 1 --dropout 0.2
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --env ${env} --run-id batchnorm --num-levels 200 --test --short --l2 0.0001 -uda 1 -norm 1
```

where all the results are including weight decay (`--l2 0.0001`) and data augmentation (`-uda 1`). 
Batchnorm is `-norm 1`, Dropout is `--dropout 0.2`, VIB is `--beta 0.0001`, L2 on Activations is
`--beta-l2a 0.0001` which corresponds to VIB-SNI with `lambda=1`. Using `--sni` switches on SNI for IBAC with `lambda=0.5`.  The number of samples for MC averages when computing the loss is set by `--nr_samples <num>` For dropout, we can either use SNI
with `lambda=0.5` by using `--sni2` or with `lambda=1.0` by using `--openai`.

Set the environment with `--env <env>`.  The number of levels in the training set is specified with `--num-levels <num>`,
and you can train on the entire level distribution by setting this flag to 0.

Using `--long` runs for 200M time steps and `--short` runs for 25M time steps and `--vshort` runs for 5M time steps.  The experiments, especially with the `--long` flag, take a while. If it's run on the VMs, it will
likely crash at some point (around 6pm is particularly likely), probably because the servers are
preemtible.
If they do, you can restart with the additional arguments `--restore-id <run-id>` and
`--restore-step <step>` where you can read out the step from the tensor-board plot.

To use try out the custom representation loss, simply add in the flag `--rep_loss`. If you want to specify a weight to this loss, pass
a number in the interval (0, 1] to `--rep_lambda`, otherwise the default weight is 1.


## Plotting

A note on the tensorboard plots: For each run, you will see 4 different folders 'name_0', 'name_1',
etc..
The 'name_0' version is the performance on the training set. The 'name_1' version is the performance
on the test set.
Furthermore, to compare to the paper you'll need to multiply the number of frames by 3, as tensorboard reports
the frames _per worker_, whereas the paper reports the total number of frames used for training.


Using `plots.py`, fill in the `path` variable, as well as `plotname`, `plotname_kl` and the
`experiments` dictionary where each entry corresponds to one line which will be the average over all
run-ids listed in the corresponding list (see the script for examples.)

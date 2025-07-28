# Population Based Training Zoo

This repo:
- contains the code for our TMLR paper [To Be Greedy, or Not to Be - That Is the Question for Population Based Training Variants](https://openreview.net/forum?id=3qmnxysNbi). 
  - **TL;DR**: Bayesian PBTs optimize the greedy objective more effectively than non-Bayesian PBTs, this can be good or bad depending on the task & the hyperparameters; no PBT variant is consistently better than others.
- serves as **the largest collection** of single-objective PBT variants (to the best of our knowledge), with all of them implementing the same interface and being applicable to arbitrary tasks, with the algorithm and task implementations decoupled. 

Algorithms: 
- [PBT](algo/pbt.py) [(paper)](https://arxiv.org/abs/1711.09846)
- [PB2](algo/pb2rand.py) [(paper)](https://arxiv.org/abs/2002.02518)
- [PB2-Mix](algo/pb2mix.py) [(paper)](https://arxiv.org/abs/2106.15883)
- [BG-PBT](algo/bgpbt.py) [(paper)](https://arxiv.org/abs/2207.09405)
- [FIRE-PBT](algo/firepbt.py) [(paper)](https://arxiv.org/abs/2109.13800)

Tasks:
- [Image classification](task/classification_amp.py)
- [Brax (reinforcement learning)](task/brax_task.py)
- Toy tasks: [Plain](task/toy_plain.py), [Deceptive](task/toy_deceptive.py)

We hope to make development & comparison of new PBT variants easier (despite a somewhat rough state of the codebase). Pull requests with new PBT variants and tasks are welcome!

## Setup

Create and activate a conda environment:

```
conda env create --file env_pbtzoo.yml
conda activate pbtzoo
```

Export PYTHONPATH and PBT_ZOO_PATH:

```
export PYTHONPATH=/path/to/this/code/dir
export PBT_ZOO_PATH=/path/to/this/code/dir
```

Start Ray:

```
ray start --head
```

#### Additional exports for Brax

Care has to be taken to manage VRAM when running several Jax processes on the same GPU. I recommend having two terminals: one where Ray is started, and one where the scripts will be run. In the Ray terminal, execute ``export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05`` prior to starting Ray; in the other terminal, execute ``export XLA_PYTHON_CLIENT_PREALLOCATE=false``.

#### Manual fix for Brax Pusher

The Pusher environment in the Brax version used in the experiments had a bug that can be fixed by manually adding a [line](https://github.com/google/brax/pull/509/files) to the Brax source code in the conda environment.

## How to use

The ``run.py`` script is the entry point. The Hydra configs are in the ``config`` directory. 

For example, PBT can be run on the CIFAR-10 dataset using the large search space (corresponds to config ``c10_20.yaml``) on the server named "star04" (``server/star04.yaml``) via the following command:

```
python run.py --config-name c10_20 server=star04 algo=pbt
```

See ``commands_tmlr.sh`` for more example commands. Note that the config numbering does not follow much of a system. For new configs, expressive names should be used instead.


## Reproducing the results of "To Be Greedy or Not to Be ..."

``commands_tmlr.sh``: commands to run the experiments.

``plot/plot_paper.py``: functions to plot the results. 

Note that the critical difference diagrams require  ``pdflatex`` to be installed and available on ``$PATH``.


## Citation

If this code was useful to you, please cite our [paper](https://openreview.net/forum?id=3qmnxysNbi). 

```
@article{
  chebykin2025to,
  title={To Be Greedy, or Not to Be {\textendash} That Is the Question for Population Based Training Variants},
  author={Alexander Chebykin and Tanja Alderliesten and Peter A. N. Bosman},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=3qmnxysNbi},
}
```

## Acknowledgments

This codebase partially incorporates code from the following repositories; we thank their authors:

- https://github.com/mirkobunse/critdd
- https://github.com/xingchenwan/bgpbt
- https://github.com/facebookresearch/how-to-autorl
- https://github.com/jparkerholder/procgen_autorl
- https://github.com/ehuynh1106/TinyImageNet-Transformers
- https://github.com/ray-project/ray
- https://github.com/learnables/learn2learn


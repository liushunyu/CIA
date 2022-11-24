# Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition

Official codebase for paper [Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition](https://arxiv.org/abs/2211.12712). This codebase is based on the open-source [PyMARL](https://github.com/oxwhirl/pymarl) framework and please refer to that repo for more documentation.



## 1. Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.

#### Install StarCraft II

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version 4.10 of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- We use the latest version SC2.4.10 for all SMAC experiments instead of SC2.4.6.2.69232.
- Performance is not comparable across versions.
```

The SMAC maps used for all experiments is in `CIA/src/envs/starcraft2/maps/SMAC_Maps` directory. You should place the `SMAC_Maps` directory in `StarCraftII/Maps`.



## 2. Usage

Please follow the instructions below to replicate the results in the paper.

#### Didactic Games: Turn
```bash
# QMIX
python src/main.py --config=qmix_turn --env-config=turn with env_args.map_name=turn

# QMIX (CIA)
python src/main.py --config=cia_grad_qmix_turn --env-config=turn with env_args.map_name=turn 
```

#### SMAC
```bash
# QMIX
python src/main.py --config=qmix_[map_name] --env-config=sc2 with env_args.map_name=[map_name]

# QPLEX
python src/main.py --config=qplex_[map_name] --env-config=sc2 with env_args.map_name=[map_name]

# QMIX (CIA)
python src/main.py --config=cia_grad_qmix_[map_name] --env-config=sc2 with env_args.map_name=[map_name] 

# QPLEX (CIA)
python src/main.py --config=cia_qplex_[map_name] --env-config=sc2 with env_args.map_name=[map_name]
```

## 3. Citation

If you find this work useful for your research, please cite our paper:

```
@article{liu2023CIA,
  title={Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition},
  author={Liu, Shunyu and Zhou, Yihe and Song, Jie and Zheng, Tongya and Chen, Kaixuan and Zhu, Tongtian and Feng, Zunlei and Song, Mingli},
  journal={AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

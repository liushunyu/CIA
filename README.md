# [AAAI 2023 Oral] Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2211.12712-b31b1b.svg)](https://arxiv.org/abs/2211.12712)

Official codebase for paper [Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition](https://arxiv.org/abs/2211.12712). This codebase is based on the open-source [PyMARL](https://github.com/oxwhirl/pymarl) framework and please refer to that repo for more documentation.

## Overview

**TLDR:** The first work identifies the ambiguous credit assignment problem in Value Decomposition (VD), a highly important ingredient for multi-agent diversity yet largely overlooked by existing literature. Therefore, we propose a novel contrastive identity-aware learning (CIA) method to promote diverse behaviors via explicitly encouraging credit-level distinguishability. The proposed CIA module imposes no constraints over the network architecture, and serves as a plug-and-play module readily applicable to various VD methods.

**Abstract:** Value Decomposition (VD) aims to deduce the contributions of agents for decentralized policies in the presence of only global rewards, and has recently emerged as a powerful credit assignment paradigm for tackling cooperative Multi-Agent Reinforcement Learning (MARL) problems. One of the main challenges in VD is to promote diverse behaviors among agents, while existing methods directly encourage the diversity of learned agent networks with various strategies. However, we argue that these dedicated designs for agent networks are still limited by the indistinguishable VD network, leading to homogeneous agent behaviors and thus downgrading the cooperation capability. In this paper, we propose a novel Contrastive Identity-Aware learning (CIA) method, explicitly boosting the credit-level distinguishability of the VD network to break the bottleneck of multi-agent diversity. Specifically, our approach leverages contrastive learning to maximize the mutual information between the temporal credits and identity representations of different agents, encouraging the full expressiveness of credit assignment and further the emergence of individualities. The algorithm implementation of the proposed CIA module is simple yet effective that can be readily incorporated into various VD architectures. Experiments on the SMAC benchmarks and across different VD backbones demonstrate that the proposed method yields results superior to the state-of-the-art counterparts.

![image](https://github.com/liushunyu/CIA/blob/master/poster.png)



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
@inproceedings{liu2023CIA,
  title={Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition},
  author={Liu, Shunyu and Zhou, Yihe and Song, Jie and Zheng, Tongya and Chen, Kaixuan and Zhu, Tongtian and Feng, Zunlei and Song, Mingli},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

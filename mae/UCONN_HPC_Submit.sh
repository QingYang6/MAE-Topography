#!/bin/bash

#SBATCH --partition=general-gpu
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=mae_initial-%j.out

/home/qiy17007/miniconda3/envs/mae/bin/python /shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/main_pretrain.py
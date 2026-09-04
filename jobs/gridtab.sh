#!/bin/bash
#SBATCH -J gridtab
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/gridtab.out
#SBATCH -e /blue/qsong1/wang.qing/systema4ST/logs/gridtab.out
#SBATCH -t 00:10:00 -c 1 --mem=4G
cd /blue/qsong1/wang.qing/systema4ST && python3 grid_table.py

#!/bin/bash
#SBATCH -J srcdata
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/srcdata.out
#SBATCH -e /blue/qsong1/wang.qing/systema4ST/logs/srcdata.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
cd /blue/qsong1/wang.qing/systema4ST && python3 sourcedata.py

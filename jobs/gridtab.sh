#!/bin/bash
#SBATCH -J gridtab
#SBATCH -o /path/to/systema4ST/logs/gridtab.out
#SBATCH -e /path/to/systema4ST/logs/gridtab.out
#SBATCH -t 00:10:00 -c 1 --mem=4G
cd /path/to/systema4ST && python3 grid_table.py

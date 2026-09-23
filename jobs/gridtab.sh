#!/bin/bash
#SBATCH -J gridtab
#SBATCH -o /path/to/project/logs/gridtab.out
#SBATCH -e /path/to/project/logs/gridtab.out
#SBATCH -t 00:10:00 -c 1 --mem=4G
cd /path/to/project && python3 grid_table.py

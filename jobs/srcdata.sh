#!/bin/bash
#SBATCH -J srcdata
#SBATCH -o /path/to/project/logs/srcdata.out
#SBATCH -e /path/to/project/logs/srcdata.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
cd /path/to/project && python3 sourcedata.py

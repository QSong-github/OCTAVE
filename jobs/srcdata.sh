#!/bin/bash
#SBATCH -J srcdata
#SBATCH -o /path/to/systema4ST/logs/srcdata.out
#SBATCH -e /path/to/systema4ST/logs/srcdata.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
cd /path/to/systema4ST && python3 sourcedata.py

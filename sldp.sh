#!/bin/bash 
#
#SBATCH --job-name=esm.msa
# Default in slurm
#SBATCH --mail-user=gz2294@cumc.columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 72:0:0 # Request 5 hours run time
# Define threads and memeory
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=100gb
# Define Output Log
date;hostname;pwd

# your script here, like python XXX, Rscript XXX
/share/vault/Users/gz2294/miniconda3/envs/GenoSCOPE/bin/python Analysis/03_2_run_sldp_analysis.py --input-file /nfs/user/Users/dl3738/BICAN/data/source/GWAS_Var/res_file.h5 --output-dir SLDP/$1 --exp-key $1 --sldp-only true --max-workers 6

date

#!/usr/bin/env python3
"""
SLDP Analysis Script
Takes an HDF5 input file and runs SLDP analysis, outputting results to a specified folder.
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import subprocess
from pyliftover import LiftOver

def main():
    parser = argparse.ArgumentParser(description='Run SLDP analysis on HDF5 input file')
    parser.add_argument('input_file', help='Path to input HDF5 file')
    parser.add_argument('output_dir', help='Output directory for SLDP results')
    parser.add_argument('--max-workers', type=int, default=12, help='Maximum number of parallel workers (default: 12)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_dir = args.output_dir
    max_workers = args.max_workers
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    print(f"Reading data from {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Read HDF5 file
    res = h5py.File(input_file, mode='r')
    
    # Extract data
    tracks = res.attrs["trial_names"]
    exp_index = res["experiments/schizophrenia/index_key"][:]
    exp_zscore = res["experiments/schizophrenia/z_score"][:]
    exp_reverse_map = res["experiments/schizophrenia/reverse_map"][:]
    all_variant_index = res["variants/index_key"][:]
    all_variant_to_pos = {val: idx for idx, val in enumerate(all_variant_index)}
    positions = [all_variant_to_pos[val] for val in exp_index]
    
    # Calculate variant effects
    exp_var_score = res["results/variant_effects"][positions, :]
    final_score = exp_var_score * (1 - 2 * exp_reverse_map).reshape(-1, 1)
    variant_effects = pd.DataFrame(final_score, columns=tracks)
    
    # Create variant and summary statistics DataFrames
    variants = pd.DataFrame({'CHR': res['variants/chr'],
                             'BP': res['variants/pos'],
                             'SNP': res['variants/rsid'], 
                             'A1': res['variants/ref'],
                             'A2': res['variants/alt'],})
    sum_stats = pd.DataFrame({'SNP': res['variants/rsid'],
                              'A1': res['variants/ref'],
                              'A2': res['variants/alt'],
                              'Z': res['experiments/schizophrenia/z_score'],
                              'N': res['experiments/schizophrenia/n_sample']})
    
    # Convert to string types
    variants['SNP'] = variants['SNP'].astype(str)
    variants['A1'] = variants['A1'].astype(str)
    variants['A2'] = variants['A2'].astype(str)
    sum_stats['SNP'] = sum_stats['SNP'].astype(str)
    sum_stats['A1'] = sum_stats['A1'].astype(str)
    sum_stats['A2'] = sum_stats['A2'].astype(str)
    variants.index = variants['SNP'].values
    sum_stats.index = sum_stats['SNP'].values
    variants['CHR'] = variants['CHR'].astype(str)
    
    # Coordinate conversion from hg38 to hg19
    print("Converting coordinates from hg38 to hg19...")
    lo = LiftOver('../Data/Ref/hg38ToHg19.over.chain.gz')
    for rsid in variants.index:
        try:
            hg19_pos = lo.convert_coordinate(variants['CHR'][rsid], variants['BP'][rsid]-1, strand='+')[0][1]
            variants.loc[rsid, 'BP'] = hg19_pos + 1  # Convert to 1-based indexing
        except IndexError:
            variants.loc[rsid, 'BP'] = np.nan  # If conversion fails, set to NaN
    
    variant_effects.index = variants.index.copy()
    
    # Read reference SNP list
    kg_rsids = pd.read_csv('SLDP/example/1000G_hm3_noMHC.rsid', header=None, names=['SNP'])
    kg_rsids['SNP'] = kg_rsids['SNP'].astype(str)
    
    # Create output directories and save variant effects
    print("Saving variant effects by track and chromosome...")
    os.makedirs(output_dir, exist_ok=True)
    for track in variant_effects.columns:
        os.makedirs(f'{output_dir}/{track}', exist_ok=True)
        track_effects = variants.copy()
        track_effects[track] = variant_effects[track].copy()
        # Drop NA 'BP' values
        track_effects = track_effects[track_effects['BP'].notna()]
        # Drop rsid not in kg_rsids
        track_effects = track_effects[track_effects['SNP'].isin(kg_rsids['SNP'])]
        # Make chromosomes numeric
        track_effects['CHR'] = track_effects['CHR'].str.replace('chr', '').astype(int)
        # Split by chromosome
        for chr in track_effects['CHR'].unique():
            # Only need SNP, A1, A2, and the track column
            chr_effects = track_effects.loc[track_effects['CHR'] == chr, ['SNP', 'A1', 'A2', track]].copy()
            # Write tab file, gzip
            chr_effects.to_csv(f'{output_dir}/{track}/{chr}.sannot.gz', sep='\t', header=True, index=False, compression='gzip')
    
    # Write summary statistics
    sum_stats.loc[track_effects.index].to_csv(f'{output_dir}/gwas.sumstats.gz', sep='\t', header=True, index=False, compression='gzip')
    
    # Define helper functions for parallel processing
    def run_sldp_preprocessing(track):
        # Check if output already exists
        if os.path.exists(f'{output_dir}/{track}/22.RV.gz'):
            cmd = f'ls -lsh {output_dir}/{track}/22.RV.gz'
        else:
            cmd = f'/share/vault/Users/gz2294/miniconda3/envs/GenoSCOPE/bin/python SLDP/sldp/preprocessannot \
                        --sannot-chr {output_dir}/{track}/ \
                        --bfile-chr SLDP/example/plink_files/1000G.EUR.QC.hm3_noMHC. \
                        --print-snps SLDP/example/1000G_hm3_noMHC.rsid \
                        --ld-blocks SLDP/example/pickrell_ldblocks.hg19.eur.bed'
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def run_sldp(track):
        # Check if output already exists
        if os.path.exists(f'{output_dir}/{track}.gwresults'):
            cmd = f'ls -lsh {output_dir}/{track}.gwresults'
        else:
            cmd = f'/share/vault/Users/gz2294/miniconda3/envs/GenoSCOPE/bin/python SLDP/sldp/sldp \
                        --pss-chr {output_dir}/gwas.KG3.95/ \
                        --sannot-chr {output_dir}/{track}/ \
                        --background-sannot-chr SLDP/example/maf5/ \
                        --outfile-stem {output_dir}/{track} \
                        --ld-blocks SLDP/example/pickrell_ldblocks.hg19.eur.bed \
                        --svd-stem SLDP/example/svds_95percent/ \
                        --bfile-reg-chr SLDP/example/plink_files/1000G.EUR.QC.hm3_noMHC. \
                        --seed 0'
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Run SLDP preprocessing in parallel
    print("Running SLDP preprocessing...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_sldp_preprocessing, track) for track in variant_effects.columns]
        for i, future in enumerate(futures):
            result = future.result()
            if result.returncode != 0:
                print(f"Error in preprocessing {variant_effects.columns[i]}: {result.stderr}")
    
    # Prepare summary stats
    print("Preparing summary statistics...")
    os.system(f'/share/vault/Users/gz2294/miniconda3/envs/SLDP/bin/python SLDP/sldp/preprocesspheno \
                --sumstats-stem {output_dir}/gwas \
                --refpanel-name SLDP/KG3.95 \
                --svd-stem SLDP/example/svds_95percent/ \
                --print-snps SLDP/example/1000G_hm3_noMHC.rsid \
                --ldscores-chr SLDP/example/LDscore/LDscore. \
                --ld-blocks SLDP/example/pickrell_ldblocks.hg19.eur.bed \
                --bfile-chr SLDP/example/plink_files/1000G.EUR.QC.hm3_noMHC.')
    
    # Run SLDP analysis in parallel
    print("Running SLDP analysis...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_sldp, track) for track in variant_effects.columns]
        for future in futures:
            result = future.result()
            if result.returncode != 0:
                print(f"Error in SLDP analysis: {result.stderr}")
    
    # Aggregate results
    print("Aggregating results...")
    final_results = pd.DataFrame()
    for track in variant_effects.columns:
        if os.path.exists(f'{output_dir}/{track}.gwresults'):
            track_res = pd.read_csv(f'{output_dir}/{track}.gwresults', sep='\t')
            final_results = pd.concat([final_results, track_res], ignore_index=True)
    final_results.index = variant_effects.columns.copy()
    
    # Save final results
    final_results.to_csv(f'{output_dir}/sldp_results.csv', sep='\t', header=True, index=True)
    
    print(f"SLDP analysis completed. Results saved to {output_dir}/sldp_results.csv")
    res.close()

if __name__ == "__main__":
    main()
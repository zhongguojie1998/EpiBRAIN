# %%
import pandas as pd
import numpy as np
import os
from pathlib import Path
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
# %%
abc = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Res/basal_ganglia_miniatlas_drop_celltype_v1/ABC_attributions_screened.tsv", sep="\t")
diffpeak = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Res/basal_ganglia_miniatlas_drop_celltype_v1/DiffPeak_attributions_screened.tsv", sep="\t")
# %%
def write_bed(abc_df, df_type='abc'):
    """
    Creates a BED-like DataFrame by combining peak coordinates with a TSS coordinate.

    Args:
        abc_df (pd.DataFrame): DataFrame containing columns 'peak_chr', 
                                'peak_start', 'peak_end', and 'pt_file'. 
                                'pt_file' is expected to have a format like 
                                'chr_start_end_...'.

    Returns:
        pd.DataFrame: A DataFrame with peak and TSS coordinates.
    """
    # 1. Create the initial DataFrame with only the peak coordinates
    bed_df = abc_df[['peak_chr', 'peak_start', 'peak_end']].copy()
    bed_df['info'] = ''
    # 2. Get TSS info from the first row of the input DataFrame
    first_row = abc_df.iloc[0]
    tss_chr = first_row['peak_chr']  # Assuming TSS is on the same chromosome
    
    # Extract start and end from the 'pt_file' string
    pt_file_parts = first_row['pt_file'].split('_')
    tss_start = int(pt_file_parts[1])
    tss_end = int(pt_file_parts[2])

    # 3. Create a new DataFrame for the single TSS row
    # The dictionary keys must match the column names of bed_df
    tss_row_df = pd.DataFrame([{
        'peak_chr': tss_chr,
        'peak_start': tss_start,
        'peak_end': tss_end
    }])
    tss_row_df['info'] = 'TSS'
    # get abc peaks
    if df_type == 'abc':
        abc_df = abc_df[abc_df['is_abc'] == True][['peak_chr', 'peak_start', 'peak_end']].copy()
        abc_df['info'] = 'ABC'
    elif df_type == 'diffpeak':
        abc_df = abc_df[abc_df['is_diffpeak'] == True][['peak_chr', 'peak_start', 'peak_end']].copy()
        abc_df['info'] = 'DiffPeak'
    else:
        raise ValueError("df_type must be either 'abc' or 'diffpeak'")
    # 4. Concatenate the original peaks DataFrame with the new TSS row DataFrame
    # ignore_index=True resets the index of the resulting DataFrame to be continuous
    combined_df = pd.concat([bed_df, tss_row_df, abc_df], ignore_index=True)
    combined_df.to_csv('test.bed', sep="\t", header=False, index=False)
    return combined_df

# %%
abc_subset = abc[abc['is_abc'] == True].copy()
diffpeak_subset = diffpeak[diffpeak['is_diffpeak'] == True].copy()
abc_subset_head = abc_subset.sort_values(by='sum_bin_score', ascending=False).head(20)
diffpeak_subset_head = diffpeak_subset.sort_values(by='sum_bin_score', ascending=False).head(20)
abc_subset_bottom = abc_subset.sort_values(by='sum_bin_score', ascending=True).head(20)
diffpeak_subset_bottom = diffpeak_subset.sort_values(by='sum_bin_score', ascending=True).head(20)
# %%

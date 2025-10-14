# %%
import h5py
import numpy as np
import pandas as pd
import polars as pl
import os
import sys
from joblib import Parallel, delayed

# Check for pyarrow (needed for reading parquet files)
try:
    import pyarrow
except ImportError:
    print("Warning: pyarrow not found. Install with: pip install pyarrow")
    print("Falling back to fastparquet if available...")

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')

# %% calculate AUPRC with SE for each dimension
def compute_auprc_with_se(trait_values, score_values, n_bootstraps=100):
    from sklearn.metrics import average_precision_score

    # Convert to polars DataFrame for efficient resampling
    V = pl.DataFrame({"label": trait_values, "score": score_values})

    # Calculate main AUPRC
    try:
        auprc = average_precision_score(trait_values, score_values)
    except ValueError:
        return np.nan, np.nan

    # Bootstrap resampling within each class
    def resample(V, seed):
        V_pos = V.filter(pl.col("label"))
        V_pos = V_pos.sample(len(V_pos), with_replacement=True, seed=seed)
        V_neg = V.filter(~pl.col("label"))
        V_neg = V_neg.sample(len(V_neg), with_replacement=True, seed=seed)
        return pl.concat([V_pos, V_neg])

    # Calculate AUPRC on bootstrap samples
    V_bs = [resample(V, i) for i in range(n_bootstraps)]
    bootstrap_auprcs = []
    for V_b in V_bs:
        try:
            bootstrap_auprcs.append(average_precision_score(V_b["label"], V_b["score"]))
        except ValueError:
            continue

    # Calculate SE as standard deviation of bootstrap estimates
    se = pl.Series(bootstrap_auprcs).std() if bootstrap_auprcs else np.nan

    return auprc, se
# %%
# Parallel computation helper functions
def compute_dim_auprc(dim, label_values, score_matrix):
    """Compute AUPRC for a single dimension (works with numpy arrays)"""
    return compute_auprc_with_se(label_values, score_matrix[:, dim])

def compute_dim_auprc_per_consequence(dim, info_df, score_matrix, consequence_categories=None):
    """Compute AUPRC for a single dimension, separated by consequence type"""
    results = {}

    # Calculate overall AUPRC
    results['overall'] = compute_auprc_with_se(info_df['label'].values, score_matrix[:, dim])

    # Calculate AUPRC for each consequence category using binary indicator columns
    if consequence_categories:
        for category in consequence_categories:
            col_name = f'is_{category}'
            if col_name in info_df.columns:
                mask = info_df[col_name]
                if mask.sum() > 0:  # Only if there are variants in this category
                    results[category] = compute_auprc_with_se(
                        info_df.loc[mask, 'label'].values,
                        score_matrix[mask, dim]
                    )

    return results

# %%
if __name__ == "__main__":
    # Loop over both mendelian and complex trait types
    for trait_type in ['mendelian', 'complex']:
        # if os.path.exists(f'Data/source/TraitGym/borzoi_{trait_type}_track_results.csv') and \
        #    os.path.exists(f'Data/source/TraitGym/bican_{trait_type}_track_results.csv'):
        #     print(f"\n{'='*80}")
        #     print(f"Skipping {trait_type} traits, results already exist.")
        #     continue
        print(f"\n{'='*80}")
        print(f"Processing {trait_type} traits...")
        print(f"{'='*80}\n")

        # %% load bican results
        bican_h5 = h5py.File(f'Data/source/TraitGym/basal_ganglia_miniatlas_drop_celltype_v1/{trait_type}_traits_test.h5', 'r')
        bican_res = bican_h5['results/log_square'][:]
        bican_info_df = pd.DataFrame({'chr': bican_h5['variants/chr'][:],
                                      'pos': bican_h5['variants/pos'][:],
                                      'ref': bican_h5['variants/ref'][:],
                                      'alt': bican_h5['variants/alt'][:]})
        bican_info_df.index = bican_info_df['chr'].astype(str) + '_' + bican_info_df['pos'].astype(str) + '_' + bican_info_df['ref'].astype(str) + '_' + bican_info_df['alt'].astype(str) + '_b38'
        bican_info_df['ID'] = bican_info_df['chr'].astype(str) + '_' + bican_info_df['pos'].astype(str) + '_' + bican_info_df['ref'].astype(str) + '_' + bican_info_df['alt'].astype(str)
        bican_res = bican_res[~bican_info_df['ID'].duplicated(keep='first')]
        bican_info_df = bican_info_df[~bican_info_df['ID'].duplicated(keep='first')]
        bican_info_df = bican_info_df.reset_index(drop=True)

        # %% get borzoi results
        borzoi_h5 = h5py.File(f'Data/source/TraitGym/borzoi_grelu/{trait_type}_traits_test.h5', 'r')
        borzoi_res = borzoi_h5['results/log_square'][:]
        borzoi_info_df = pd.DataFrame({'chr': borzoi_h5['variants/chr'][:],
                                        'pos': borzoi_h5['variants/pos'][:],
                                        'ref': borzoi_h5['variants/ref'][:],
                                        'alt': borzoi_h5['variants/alt'][:]})
        borzoi_info_df.index = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str) + '_b38'
        borzoi_info_df['ID'] = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str)
        borzoi_res = borzoi_res[~borzoi_info_df['ID'].duplicated(keep='first')]
        borzoi_info_df = borzoi_info_df[~borzoi_info_df['ID'].duplicated(keep='first')]
        borzoi_info_df = borzoi_info_df.reset_index(drop=True)

        # %% read in TraitGym files
        trait_info = pd.read_csv(f'Data/source/TraitGym/{trait_type}_traits_test.csv', sep=',')

        # %% merge trait_info with borzoi_info_df
        trait_info['ID'] = 'chr' + trait_info['chrom'].astype(str) + '_' + trait_info['pos'].astype(str) + '_' + trait_info['ref'].astype(str) + '_' + trait_info['alt'].astype(str)
        borzoi_info_df = pd.merge(borzoi_info_df, trait_info, left_on='ID', right_on='ID', how='inner')
        bican_info_df = pd.merge(bican_info_df, trait_info, left_on='ID', right_on='ID', how='inner')

        # %% Define consequence categories and load variant subsets
        # Different categories and paths for mendelian vs complex traits
        if trait_type == 'complex':
            consequence_categories = [
                'nonexonic_AND_distal',
                'nonexonic_AND_proximal',
                '5_prime_UTR_variant',
                '3_prime_UTR_variant',
                'non_coding_transcript_exon_variant',
                'disease',
                'non_disease'
            ]
            base_path = '/gpfs/commons/groups/ren_lab/guojiezhong/TraitGym/results/dataset/complex_traits_matched_9/subset'
        elif trait_type == 'mendelian':
            consequence_categories = [
                'nonexonic_AND_distal',
                'nonexonic_AND_proximal',
                '5_prime_UTR_variant',
                '3_prime_UTR_variant',
                'non_coding_transcript_exon_variant'
            ]
            base_path = '/gpfs/commons/groups/ren_lab/guojiezhong/TraitGym/results/dataset/mendelian_traits_matched_9/subset'
        else:
            print(f"  Unknown trait type: {trait_type}")
            consequence_categories = []
            base_path = None

        if consequence_categories:
            # Read all category files and create binary indicator columns for each category
            # This allows variants to belong to multiple categories
            print(f"  Loading consequence categories from parquet files...")
            print(f"  Base path: {base_path}")

            # Initialize all category columns to False
            for category in consequence_categories:
                borzoi_info_df[f'is_{category}'] = False
                bican_info_df[f'is_{category}'] = False

            # Load each category and mark variants
            for category in consequence_categories:
                parquet_file = f'{base_path}/{category}.parquet'
                try:
                    category_df = pd.read_parquet(parquet_file)
                    # Create variant ID
                    category_df['ID'] = 'chr' + category_df['chrom'].astype(str) + '_' + category_df['pos'].astype(str) + '_' + category_df['ref'].astype(str) + '_' + category_df['alt'].astype(str)
                    category_ids = set(category_df['ID'])

                    # Mark variants that belong to this category
                    borzoi_info_df[f'is_{category}'] = borzoi_info_df['ID'].isin(category_ids)
                    bican_info_df[f'is_{category}'] = bican_info_df['ID'].isin(category_ids)

                    n_borzoi = borzoi_info_df[f'is_{category}'].sum()
                    n_bican = bican_info_df[f'is_{category}'].sum()
                    print(f"    - {category}: {len(category_df)} total variants, {n_borzoi} in Borzoi, {n_bican} in BICAN")
                except Exception as e:
                    print(f"    - {category}: Error loading file - {e}")

            # Print statistics about category overlap
            print(f"\n  Category membership statistics:")

            # Count how many categories each variant belongs to
            borzoi_category_cols = [f'is_{cat}' for cat in consequence_categories]
            bican_category_cols = [f'is_{cat}' for cat in consequence_categories]

            borzoi_n_categories = borzoi_info_df[borzoi_category_cols].sum(axis=1)
            bican_n_categories = bican_info_df[bican_category_cols].sum(axis=1)

            print(f"\n  Borzoi variants by number of categories:")
            print(f"    0 categories (unassigned): {(borzoi_n_categories == 0).sum()} ({(borzoi_n_categories == 0).sum()/len(borzoi_info_df)*100:.1f}%)")
            print(f"    1 category: {(borzoi_n_categories == 1).sum()} ({(borzoi_n_categories == 1).sum()/len(borzoi_info_df)*100:.1f}%)")
            if (borzoi_n_categories > 1).any():
                print(f"    2+ categories: {(borzoi_n_categories > 1).sum()} ({(borzoi_n_categories > 1).sum()/len(borzoi_info_df)*100:.1f}%)")
                print(f"    Max categories per variant: {borzoi_n_categories.max()}")

            print(f"\n  BICAN variants by number of categories:")
            print(f"    0 categories (unassigned): {(bican_n_categories == 0).sum()} ({(bican_n_categories == 0).sum()/len(bican_info_df)*100:.1f}%)")
            print(f"    1 category: {(bican_n_categories == 1).sum()} ({(bican_n_categories == 1).sum()/len(bican_info_df)*100:.1f}%)")
            if (bican_n_categories > 1).any():
                print(f"    2+ categories: {(bican_n_categories > 1).sum()} ({(bican_n_categories > 1).sum()/len(bican_info_df)*100:.1f}%)")
                print(f"    Max categories per variant: {bican_n_categories.max()}")

            # Optional: Filter to only variants with assigned consequences
            # Uncomment the lines below to exclude variants not found in any category file
            # This will remove variants where consequence is NaN/None
            # borzoi_mask = borzoi_info_df['consequence'].notna()
            # bican_mask = bican_info_df['consequence'].notna()
            # borzoi_info_df = borzoi_info_df[borzoi_mask].reset_index(drop=True)
            # bican_info_df = bican_info_df[bican_mask].reset_index(drop=True)
            # borzoi_res = borzoi_res[borzoi_mask]
            # bican_res = bican_res[bican_mask]
            # print(f"\n  After filtering: {len(borzoi_info_df)} Borzoi variants, {len(bican_info_df)} BICAN variants")

        # %% read in track annotations
        borzoi_track_anno = pd.read_csv('Data/data_config/borzoi.published.targets.csv', sep=',', index_col=0)
        bican_track_anno = pd.read_csv('logs/basal_ganglia_miniatlas_drop_celltype_v1/regression_label_meta.csv', sep=',')

        # %% Parallel AUPRC calculation with per-consequence breakdown
        # Determine which consequence categories to use based on trait type
        active_consequence_categories = consequence_categories if 'consequence_categories' in locals() else None

        print(f"Calculating AUPRC per dimension for Borzoi (parallel, {borzoi_res.shape[1]} dimensions)...")
        borzoi_results_detailed = Parallel(n_jobs=36, backend='loky', verbose=10)(
            delayed(compute_dim_auprc_per_consequence)(dim, borzoi_info_df, borzoi_res, active_consequence_categories)
            for dim in range(borzoi_res.shape[1])
        )

        # Extract overall AUPRC
        borzoi_auprc = [r['overall'][0] for r in borzoi_results_detailed]
        borzoi_auprc_se = [r['overall'][1] for r in borzoi_results_detailed]

        # Append overall to borzoi_track_anno
        borzoi_track_anno['AUPRC'] = borzoi_auprc
        borzoi_track_anno['AUPRC_SE'] = borzoi_auprc_se

        # Extract and append per-consequence AUPRC
        if active_consequence_categories:
            print(f"  Adding per-consequence AUPRC for {len(active_consequence_categories)} categories")
            for consequence in active_consequence_categories:
                auprc_list = []
                se_list = []
                for r in borzoi_results_detailed:
                    if consequence in r:
                        auprc_list.append(r[consequence][0])
                        se_list.append(r[consequence][1])
                    else:
                        auprc_list.append(np.nan)
                        se_list.append(np.nan)
                borzoi_track_anno[f'AUPRC_{consequence}'] = auprc_list
                borzoi_track_anno[f'AUPRC_SE_{consequence}'] = se_list

        print(f"Calculating AUPRC per dimension for BICAN (parallel, {bican_res.shape[1]} dimensions)...")
        bican_results_detailed = Parallel(n_jobs=36, backend='loky', verbose=10)(
            delayed(compute_dim_auprc_per_consequence)(dim, bican_info_df, bican_res, active_consequence_categories)
            for dim in range(bican_res.shape[1])
        )

        # Extract overall AUPRC
        bican_auprc = [r['overall'][0] for r in bican_results_detailed]
        bican_auprc_se = [r['overall'][1] for r in bican_results_detailed]

        # Append overall to bican_track_anno
        bican_track_anno['AUPRC'] = bican_auprc
        bican_track_anno['AUPRC_SE'] = bican_auprc_se

        # Extract and append per-consequence AUPRC
        if active_consequence_categories:
            for consequence in active_consequence_categories:
                auprc_list = []
                se_list = []
                for r in bican_results_detailed:
                    if consequence in r:
                        auprc_list.append(r[consequence][0])
                        se_list.append(r[consequence][1])
                    else:
                        auprc_list.append(np.nan)
                        se_list.append(np.nan)
                bican_track_anno[f'AUPRC_{consequence}'] = auprc_list
                bican_track_anno[f'AUPRC_SE_{consequence}'] = se_list

        # %% Calculate modality-level and cell_type-level aggregated AUPRC
        print(f"\nCalculating modality-level and cell_type-level aggregated AUPRC...")

        # Add modality information to track annotations
        borzoi_track_anno['modality'] = borzoi_track_anno['file'].str.split('/').str[6]
        # For BICAN, modality is already in the annotation file
        # (no need to modify, it's already there from the CSV)

        # Add cell_type information
        # For Borzoi, cell_type is the same as modality (no cell type specificity)
        borzoi_track_anno['cell_type'] = borzoi_track_anno['modality']
        # For BICAN, cell_type is already in the annotation file from the CSV

        # Function to compute modality-level AUPRC
        def compute_modality_auprc(modality_tracks, score_matrix, info_df, consequence_categories=None):
            """Aggregate scores across tracks in a modality and compute AUPRC"""
            # L2 aggregate scores across tracks
            aggregated_scores = np.linalg.norm(score_matrix[:, modality_tracks], axis=1)

            results = {}
            # Overall AUPRC
            results['overall'] = compute_auprc_with_se(info_df['label'].values, aggregated_scores)

            # Per-consequence AUPRC
            if consequence_categories:
                for category in consequence_categories:
                    col_name = f'is_{category}'
                    if col_name in info_df.columns:
                        mask = info_df[col_name]
                        if mask.sum() > 0:
                            results[category] = compute_auprc_with_se(
                                info_df.loc[mask, 'label'].values,
                                aggregated_scores[mask]
                            )
            return results

        # Compute for Borzoi modalities
        borzoi_modality_results = []
        borzoi_modalities = borzoi_track_anno['modality'].unique()
        print(f"  Processing {len(borzoi_modalities)} Borzoi modalities...")

        for modality in borzoi_modalities:
            if pd.isna(modality):
                continue
            track_indices = borzoi_track_anno[borzoi_track_anno['modality'] == modality].index.tolist()
            if len(track_indices) > 0:
                results = compute_modality_auprc(track_indices, borzoi_res, borzoi_info_df, active_consequence_categories)
                result_dict = {
                    'modality': modality,
                    'n_tracks': len(track_indices),
                    'AUPRC': results['overall'][0],
                    'AUPRC_SE': results['overall'][1]
                }
                # Add per-consequence results
                if active_consequence_categories:
                    for category in active_consequence_categories:
                        if category in results:
                            result_dict[f'AUPRC_{category}'] = results[category][0]
                            result_dict[f'AUPRC_SE_{category}'] = results[category][1]
                borzoi_modality_results.append(result_dict)
                print(f"    - {modality}: {len(track_indices)} tracks, AUPRC={results['overall'][0]:.4f}±{results['overall'][1]:.4f}")

        borzoi_modality_df = pd.DataFrame(borzoi_modality_results)
        borzoi_modality_df['model'] = 'Borzoi'

        # Compute for BICAN modalities
        bican_modality_results = []
        bican_modalities = bican_track_anno['modality'].unique()
        print(f"  Processing {len(bican_modalities)} BICAN modalities...")

        for modality in bican_modalities:
            if pd.isna(modality):
                continue
            track_indices = bican_track_anno[bican_track_anno['modality'] == modality].index.tolist()
            if len(track_indices) > 0:
                results = compute_modality_auprc(track_indices, bican_res, bican_info_df, active_consequence_categories)
                result_dict = {
                    'modality': modality,
                    'n_tracks': len(track_indices),
                    'AUPRC': results['overall'][0],
                    'AUPRC_SE': results['overall'][1]
                }
                # Add per-consequence results
                if active_consequence_categories:
                    for category in active_consequence_categories:
                        if category in results:
                            result_dict[f'AUPRC_{category}'] = results[category][0]
                            result_dict[f'AUPRC_SE_{category}'] = results[category][1]
                bican_modality_results.append(result_dict)
                print(f"    - {modality}: {len(track_indices)} tracks, AUPRC={results['overall'][0]:.4f}±{results['overall'][1]:.4f}")

        bican_modality_df = pd.DataFrame(bican_modality_results)
        bican_modality_df['model'] = 'BICAN'

        # Combine modality results
        modality_results_df = pd.concat([borzoi_modality_df, bican_modality_df], axis=0, ignore_index=True)

        # %% Compute cell_type-level aggregated AUPRC
        print(f"\n  Computing cell_type-level aggregated AUPRC...")

        # Compute for Borzoi cell_types (same as modalities for Borzoi)
        borzoi_celltype_results = []
        borzoi_celltypes = borzoi_track_anno['cell_type'].unique()
        print(f"  Processing {len(borzoi_celltypes)} Borzoi cell types...")

        for cell_type in borzoi_celltypes:
            if pd.isna(cell_type):
                continue
            track_indices = borzoi_track_anno[borzoi_track_anno['cell_type'] == cell_type].index.tolist()
            if len(track_indices) > 0:
                results = compute_modality_auprc(track_indices, borzoi_res, borzoi_info_df, active_consequence_categories)
                result_dict = {
                    'cell_type': cell_type,
                    'n_tracks': len(track_indices),
                    'AUPRC': results['overall'][0],
                    'AUPRC_SE': results['overall'][1]
                }
                # Add per-consequence results
                if active_consequence_categories:
                    for category in active_consequence_categories:
                        if category in results:
                            result_dict[f'AUPRC_{category}'] = results[category][0]
                            result_dict[f'AUPRC_SE_{category}'] = results[category][1]
                borzoi_celltype_results.append(result_dict)
                print(f"    - {cell_type}: {len(track_indices)} tracks, AUPRC={results['overall'][0]:.4f}±{results['overall'][1]:.4f}")

        borzoi_celltype_df = pd.DataFrame(borzoi_celltype_results)
        borzoi_celltype_df['model'] = 'Borzoi'

        # Compute for BICAN cell_types
        bican_celltype_results = []
        bican_celltypes = bican_track_anno['cell_type'].unique()
        print(f"  Processing {len(bican_celltypes)} BICAN cell types...")

        for cell_type in bican_celltypes:
            if pd.isna(cell_type):
                continue
            # only use tracks with this cell type and modalities without RNA
            track_indices = bican_track_anno[(bican_track_anno['cell_type'] == cell_type) & (bican_track_anno['modality'].str.contains('RNA') == False)].index.tolist()
            if len(track_indices) > 0:
                results = compute_modality_auprc(track_indices, bican_res, bican_info_df, active_consequence_categories)
                result_dict = {
                    'cell_type': cell_type,
                    'n_tracks': len(track_indices),
                    'AUPRC': results['overall'][0],
                    'AUPRC_SE': results['overall'][1]
                }
                # Add per-consequence results
                if active_consequence_categories:
                    for category in active_consequence_categories:
                        if category in results:
                            result_dict[f'AUPRC_{category}'] = results[category][0]
                            result_dict[f'AUPRC_SE_{category}'] = results[category][1]
                bican_celltype_results.append(result_dict)
                print(f"    - {cell_type}: {len(track_indices)} tracks, AUPRC={results['overall'][0]:.4f}±{results['overall'][1]:.4f}")

        bican_celltype_df = pd.DataFrame(bican_celltype_results)
        bican_celltype_df['model'] = 'BICAN'

        # Combine cell_type results
        celltype_results_df = pd.concat([borzoi_celltype_df, bican_celltype_df], axis=0, ignore_index=True)

        # %% save results
        borzoi_track_anno.to_csv(f'Data/source/TraitGym/borzoi_{trait_type}_track_results.csv')
        bican_track_anno.to_csv(f'Data/source/TraitGym/bican_{trait_type}_track_results.csv')
        modality_results_df.to_csv(f'Data/source/TraitGym/modality_{trait_type}_results.csv', index=False)
        celltype_results_df.to_csv(f'Data/source/TraitGym/celltype_{trait_type}_results.csv', index=False)

        print(f"\nDone with {trait_type} traits! Results saved.")
        print(f"  - Data/source/TraitGym/borzoi_{trait_type}_track_results.csv")
        print(f"  - Data/source/TraitGym/bican_{trait_type}_track_results.csv")
        print(f"  - Data/source/TraitGym/modality_{trait_type}_results.csv")
        print(f"  - Data/source/TraitGym/celltype_{trait_type}_results.csv")

        # Print summary of columns added
        if active_consequence_categories:
            consequence_cols = [col for col in borzoi_track_anno.columns if col.startswith('AUPRC_') and col != 'AUPRC_SE' and not col.endswith('_SE')]
            print(f"\n  Added per-consequence AUPRC columns:")
            for col in consequence_cols:
                if col != 'AUPRC':  # Skip the overall AUPRC
                    print(f"    - {col}")
        # %% plot different tracks against borzoi
        borzoi_track_anno = pd.read_csv(f'Data/source/TraitGym/borzoi_{trait_type}_track_results.csv', index_col=0)
        bican_track_anno = pd.read_csv(f'Data/source/TraitGym/bican_{trait_type}_track_results.csv', index_col=0)
        modality_results_df = pd.read_csv(f'Data/source/TraitGym/modality_{trait_type}_results.csv')
        celltype_results_df = pd.read_csv(f'Data/source/TraitGym/celltype_{trait_type}_results.csv')
        # Note: modality info already added above, just need to set model column for plotting
        borzoi_track_anno['model'] = 'Borzoi'
        bican_track_anno['model'] = 'BICAN'
        track_anno_merged = pd.concat([borzoi_track_anno, bican_track_anno], axis=0, ignore_index=True)
        # %% violin plot
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Plot overall AUPRC
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=track_anno_merged, x='modality', y='AUPRC', hue='model', split=False, inner='quartile', ax=ax)
        ax.set_title(f'Overall AUPRC - {trait_type.capitalize()} Traits')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'Data/source/TraitGym/{trait_type}_overall_AUPRC_violin.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot per-consequence AUPRC (find all AUPRC columns that are not overall or SE)
        auprc_consequence_cols = [col for col in track_anno_merged.columns
                                   if col.startswith('AUPRC_') and not '_SE' in col]

        if auprc_consequence_cols:
            # Melt the dataframe to have consequence as a variable
            id_vars = ['model', 'modality']
            # Only keep columns that exist in the dataframe
            id_vars = [col for col in id_vars if col in track_anno_merged.columns]

            melted_df = track_anno_merged[id_vars + auprc_consequence_cols].melt(
                id_vars=id_vars,
                value_vars=auprc_consequence_cols,
                var_name='consequence',
                value_name='AUPRC'
            )

            # Clean up consequence names (remove 'AUPRC_' prefix)
            melted_df['consequence'] = melted_df['consequence'].str.replace('AUPRC_', '')

            # Create combined model+modality column for x-axis
            melted_df['model_modality'] = melted_df['model'] + ' - ' + melted_df['modality'].astype(str)

            # Create separate violin plot for each consequence type
            consequences = melted_df['consequence'].unique()
            print(f"  Creating plots for {len(consequences)} consequence types...")

            for consequence in consequences:
                consequence_df = melted_df[melted_df['consequence'] == consequence]

                fig, ax = plt.subplots(figsize=(16, 6))
                sns.violinplot(data=consequence_df, x='model_modality', y='AUPRC', hue='model',
                              split=False, inner='quartile', ax=ax)
                ax.set_title(f'AUPRC for {consequence} - {trait_type.capitalize()} Traits')
                ax.legend(title='Model')
                ax.set_ylabel('AUPRC')
                ax.set_xlabel('Model - Modality')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                # Create safe filename
                safe_consequence = consequence.replace('/', '_').replace(' ', '_')
                plt.savefig(f'Data/source/TraitGym/{trait_type}_{safe_consequence}_AUPRC_violin.png', dpi=300, bbox_inches='tight')
                plt.close()

            print(f"  Created plots:")
            print(f"    - Data/source/TraitGym/{trait_type}_overall_AUPRC_violin.png")
            print(f"    - Data/source/TraitGym/{trait_type}_{{consequence}}_AUPRC_violin.png (x{len(consequences)} consequences)")

        # %% Plot modality-level aggregated results
        print(f"\n  Creating modality-level aggregated AUPRC plots...")

        # Bar plot for overall modality AUPRC with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        modality_plot_df = modality_results_df.copy()
        modality_plot_df['modality_model'] = modality_plot_df['modality'].astype(str) + '\n(' + modality_plot_df['model'] + ')'

        x_pos = np.arange(len(modality_plot_df))
        colors = ['#1f77b4' if m == 'Borzoi' else '#ff7f0e' for m in modality_plot_df['model']]

        ax.bar(x_pos, modality_plot_df['AUPRC'], yerr=modality_plot_df['AUPRC_SE'],
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(modality_plot_df['modality_model'], rotation=45, ha='right')
        ax.set_ylabel('AUPRC')
        ax.set_title(f'Modality-Level Aggregated AUPRC - {trait_type.capitalize()} Traits')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Borzoi'),
                          Patch(facecolor='#ff7f0e', edgecolor='black', label='BICAN')]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(f'Data/source/TraitGym/{trait_type}_modality_aggregated_AUPRC.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot per-consequence modality AUPRC
        auprc_modality_consequence_cols = [col for col in modality_results_df.columns
                                           if col.startswith('AUPRC_') and not '_SE' in col]

        if auprc_modality_consequence_cols:
            # Create separate bar plot for each consequence type
            for consequence_col in auprc_modality_consequence_cols:
                se_col = consequence_col.replace('AUPRC_', 'AUPRC_SE_')
                if se_col not in modality_results_df.columns:
                    continue

                consequence_name = consequence_col.replace('AUPRC_', '')
                plot_df = modality_results_df[['modality', 'model', consequence_col, se_col]].dropna()

                if len(plot_df) == 0:
                    continue

                plot_df['modality_model'] = plot_df['modality'].astype(str) + '\n(' + plot_df['model'] + ')'

                fig, ax = plt.subplots(figsize=(10, 6))
                x_pos = np.arange(len(plot_df))
                colors = ['#1f77b4' if m == 'Borzoi' else '#ff7f0e' for m in plot_df['model']]

                ax.bar(x_pos, plot_df[consequence_col], yerr=plot_df[se_col],
                       capsize=5, color=colors, alpha=0.7, edgecolor='black')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(plot_df['modality_model'], rotation=45, ha='right')
                ax.set_ylabel('AUPRC')
                ax.set_title(f'Modality-Level AUPRC for {consequence_name} - {trait_type.capitalize()} Traits')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                # Add legend
                ax.legend(handles=legend_elements)

                plt.tight_layout()
                safe_consequence = consequence_name.replace('/', '_').replace(' ', '_')
                plt.savefig(f'Data/source/TraitGym/{trait_type}_modality_{safe_consequence}_AUPRC.png', dpi=300, bbox_inches='tight')
                plt.close()

            print(f"  Created modality plots:")
            print(f"    - Data/source/TraitGym/{trait_type}_modality_aggregated_AUPRC.png")
            print(f"    - Data/source/TraitGym/{trait_type}_modality_{{consequence}}_AUPRC.png (x{len(auprc_modality_consequence_cols)} consequences)")

        # %% Plot cell_type-level aggregated results
        print(f"\n  Creating cell_type-level aggregated AUPRC plots...")

        # Bar plot for overall cell_type AUPRC with error bars
        fig, ax = plt.subplots(figsize=(16, 6))
        celltype_plot_df = celltype_results_df.copy()
        celltype_plot_df['celltype_model'] = celltype_plot_df['cell_type'].astype(str) + '\n(' + celltype_plot_df['model'] + ')'

        x_pos = np.arange(len(celltype_plot_df))
        colors = ['#1f77b4' if m == 'Borzoi' else '#ff7f0e' for m in celltype_plot_df['model']]

        ax.bar(x_pos, celltype_plot_df['AUPRC'], yerr=celltype_plot_df['AUPRC_SE'],
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(celltype_plot_df['celltype_model'], rotation=90, ha='right', fontsize=6)
        ax.set_ylabel('AUPRC')
        ax.set_title(f'Cell Type-Level Aggregated AUPRC - {trait_type.capitalize()} Traits')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Borzoi'),
                          Patch(facecolor='#ff7f0e', edgecolor='black', label='BICAN')]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(f'Data/source/TraitGym/{trait_type}_celltype_aggregated_AUPRC.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot per-consequence cell_type AUPRC
        auprc_celltype_consequence_cols = [col for col in celltype_results_df.columns
                                           if col.startswith('AUPRC_') and not '_SE' in col]

        if auprc_celltype_consequence_cols:
            # Create separate bar plot for each consequence type
            for consequence_col in auprc_celltype_consequence_cols:
                se_col = consequence_col.replace('AUPRC_', 'AUPRC_SE_')
                if se_col not in celltype_results_df.columns:
                    continue

                consequence_name = consequence_col.replace('AUPRC_', '')
                plot_df = celltype_results_df[['cell_type', 'model', consequence_col, se_col]].dropna()

                if len(plot_df) == 0:
                    continue

                plot_df['celltype_model'] = plot_df['cell_type'].astype(str) + '\n(' + plot_df['model'] + ')'

                fig, ax = plt.subplots(figsize=(16, 6))
                x_pos = np.arange(len(plot_df))
                colors = ['#1f77b4' if m == 'Borzoi' else '#ff7f0e' for m in plot_df['model']]

                ax.bar(x_pos, plot_df[consequence_col], yerr=plot_df[se_col],
                       capsize=5, color=colors, alpha=0.7, edgecolor='black')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(plot_df['celltype_model'], rotation=90, ha='right', fontsize=6)
                ax.set_ylabel('AUPRC')
                ax.set_title(f'Cell Type-Level AUPRC for {consequence_name} - {trait_type.capitalize()} Traits')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                # Add legend
                ax.legend(handles=legend_elements)

                plt.tight_layout()
                safe_consequence = consequence_name.replace('/', '_').replace(' ', '_')
                plt.savefig(f'Data/source/TraitGym/{trait_type}_celltype_{safe_consequence}_AUPRC.png', dpi=300, bbox_inches='tight')
                plt.close()

            print(f"  Created cell_type plots:")
            print(f"    - Data/source/TraitGym/{trait_type}_celltype_aggregated_AUPRC.png")
            print(f"    - Data/source/TraitGym/{trait_type}_celltype_{{consequence}}_AUPRC.png (x{len(auprc_celltype_consequence_cols)} consequences)")
    # %%
    print(f"\n{'='*80}")
    print("All trait types processed successfully!")
    print(f"{'='*80}")
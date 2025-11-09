import logging
import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
import sys

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from utils.logging import BaseLogger


def process_celltype_cutoffs(celltype, input_file, ccre_bed_df, output_dir, cutoffs=[2, 5, 10]):
    """
    Process a single cell type and generate filtered BED files for different cutoffs.

    Args:
        celltype: Cell type name
        input_file: Path to the cCRE-gene attribution CSV file
        ccre_bed_df: DataFrame with full cCRE BED information
        output_dir: Output directory for filtered BED files
        cutoffs: List of cutoff values to apply

    Returns:
        dict: Summary statistics for this cell type
    """
    logger = BaseLogger(name=f"FilterCCRE_{celltype}", level=logging.INFO)

    # Read the attribution file
    logger.info(f"Reading {input_file}...")
    attribution_df = pd.read_csv(input_file, index_col=0)

    # Get maximum value for each row (cCRE), ignoring NaN values
    max_values = attribution_df.max(axis=1, skipna=True)

    results = {}
    for cutoff in cutoffs:
        # Get cCRE names that pass the cutoff
        passing_ccres = max_values[max_values >= cutoff].index.tolist()

        # Subset the BED file
        filtered_bed = ccre_bed_df[ccre_bed_df['Name'].isin(passing_ccres)]

        # Save to file
        output_file = os.path.join(output_dir, f"{celltype}_{cutoff}.bed")
        filtered_bed.to_csv(output_file, sep='\t', header=False, index=False)

        results[cutoff] = len(passing_ccres)
        logger.info(f"  Cutoff {cutoff}: {len(passing_ccres)} cCREs -> {output_file}")

    return celltype, results


def main():
    # Setup
    logger = BaseLogger(name="FilterCCRE", level=logging.INFO)

    # Paths
    ccre_bed_path = "Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed"
    input_dir = "Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/DiffExpress_cCRE"
    output_dir = "Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/DiffExpress_cCRE_filtered"

    os.makedirs(output_dir, exist_ok=True)

    # Load cCRE bed file
    logger.info(f"Loading cCRE bed file from {ccre_bed_path}...")
    ccre_bed_df = pd.read_csv(
        ccre_bed_path,
        sep='\t',
        header=None,
        names=['Chromosome', 'Start', 'End', 'Name', 'Score']
    )
    logger.info(f"Loaded {len(ccre_bed_df)} cCREs")

    # Get all cell type attribution files
    logger.info(f"Scanning files in {input_dir}...")
    attribution_files = [f for f in os.listdir(input_dir) if f.endswith('_cCRE_gene_attribution.csv')]
    logger.info(f"Found {len(attribution_files)} attribution files")

    # Extract cell type names
    celltypes = [f.replace('_cCRE_gene_attribution.csv', '') for f in attribution_files]

    # Prepare arguments for parallel processing
    cutoffs = [1, 2, 5, 10]
    args_list = [
        (
            celltype,
            os.path.join(input_dir, f"{celltype}_cCRE_gene_attribution.csv"),
            ccre_bed_df,
            output_dir,
            cutoffs
        )
        for celltype in celltypes
    ]

    # Process cell types in parallel
    logger.info(f"\nProcessing {len(celltypes)} cell types in parallel...")
    logger.info(f"Cutoffs: {cutoffs}")

    results = Parallel(n_jobs=36, verbose=10)(
        delayed(process_celltype_cutoffs)(*args) for args in args_list
    )

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Summary of filtered cCREs by cutoff")
    logger.info("="*80)

    # Organize results by cutoff
    cutoff_summary = {cutoff: [] for cutoff in cutoffs}
    for celltype, celltype_results in results:
        for cutoff, count in celltype_results.items():
            cutoff_summary[cutoff].append((celltype, count))

    for cutoff in cutoffs:
        logger.info(f"\nCutoff >= {cutoff}:")
        for celltype, count in sorted(cutoff_summary[cutoff], key=lambda x: x[1], reverse=True):
            logger.info(f"  {celltype}: {count} cCREs")

    logger.info("\nProcessing complete!")


if __name__ == "__main__":
    # Add parent directory to path
    import sys
    ROOT = Path(__file__).parent.parent
    sys.path.append(str(ROOT / "Model"))
    os.chdir(ROOT)

    main()

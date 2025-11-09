import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pyranges as pr
import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from data.tokenizer import FastaInterval
from utils.config import load_config
from utils.logging import BaseLogger


def parse_filename(filename):
    """
    Parse filename to extract metadata.
    Format: chr_start_end_gene_celltype_strand_other-22_random.pt
    Example: chr10_100047062_100571350_PKD2L1_L45IT_plus_other-22_random.pt

    Returns:
        dict with keys: chr, start, end, gene, celltype, strand
    """
    parts = filename.replace('.pt', '').split('_')

    # Find the position of 'other' which is always followed by '-22'
    # Work backwards from the end
    # Last parts: ..._strand_other-22_random

    # The structure is: chr_start_end_gene_celltype_strand_other-22_random
    # So we can work backwards:
    # -1: random
    # -2: other-22
    # -3: strand (plus/minus)
    # -4: celltype
    # -5+: gene (may have multiple parts)

    chr_name = parts[0]
    start = int(parts[1])
    end = int(parts[2])

    # Strand is always 3rd from last (before other-22 and random)
    strand = parts[-3]
    celltype = parts[-4]

    # Gene name is everything between index 3 and celltype position
    gene = '_'.join(parts[3:-4])

    return {
        'chr': chr_name,
        'start': start,
        'end': end,
        'gene': gene,
        'celltype': celltype,
        'strand': strand,
        'filename': filename
    }


def calculate_ccre_attribution(
    pt_file_path,
    chr_name,
    start,
    end,
    ccre_ranges,
    dna_tokenizer,
    logger
):
    """
    Calculate attribution scores for cCREs overlapping with the sequence region.

    Args:
        pt_file_path: Path to the .pt file containing attribution tensor
        chr_name: Chromosome name
        start: Sequence start position
        end: Sequence end position
        ccre_ranges: PyRanges object containing cCREs
        dna_tokenizer: FastaInterval tokenizer
        logger: Logger object

    Returns:
        dict: {ccre_name: attribution_score}
    """
    try:
        # Load attribution tensor [1, seq_len, 4]
        attribution = torch.load(pt_file_path, weights_only=False)

        # Get onehot encoding of the sequence
        token_dict = dna_tokenizer(
            chr_name=chr_name,
            start=start,
            end=end,
            return_augs=False,
            return_rela_idx=True
        )

        test_seq_onehot = token_dict["one_hot"]  # [seq_len, 4]
        real_start, real_end = token_dict["real_region"]

        # Multiply attribution by onehot and sum over nucleotide dimension
        # attribution: [1, seq_len, 4], test_seq_onehot: [seq_len, 4]
        with torch.no_grad():
            # Element-wise multiplication and sum over last dimension
            signal = (attribution * test_seq_onehot.unsqueeze(0)).sum(dim=-1)  # [1, seq_len]
            signal = signal.squeeze(0)  # [seq_len]

        # Find overlapping cCREs
        seq_range = pr.PyRanges(chromosomes=[chr_name], starts=[real_start], ends=[real_end])
        overlapping = ccre_ranges.join(seq_range)

        if len(overlapping) == 0:
            return {}

        # Calculate attribution for each overlapping cCRE
        ccre_scores = {}

        for idx in range(len(overlapping)):
            ccre_start = overlapping.Start[idx]
            ccre_end = overlapping.End[idx]
            ccre_name = overlapping.Name[idx]

            # Calculate relative indices within the sequence
            # The sequence starts at real_start and has length seq_len
            rel_start = max(0, ccre_start - real_start)
            rel_end = min(signal.shape[0], ccre_end - real_start)

            # Extract and sum attribution scores for this cCRE
            if rel_start < rel_end and rel_start < signal.shape[0] and rel_end > 0 and (rel_end-rel_start)==(ccre_end - ccre_start):
                ccre_attribution = signal[rel_start:rel_end].sum().item()
                ccre_scores[ccre_name] = ccre_attribution

        return ccre_scores

    except Exception as e:
        logger.error(f"Error processing {pt_file_path}: {str(e)}")
        return {}


def process_celltype(
    celltype,
    celltype_files,
    ccre_df,
    ccre_ranges,
    dna_tokenizer,
    interp_diff_dir,
    output_dir,
    fasta_file,
    context_length
):
    """
    Process a single cell type and generate its cCRE-gene attribution dataframe.

    Args:
        celltype: Cell type name
        celltype_files: DataFrame with metadata for files of this cell type
        ccre_df: DataFrame with cCRE information
        ccre_ranges: PyRanges object with cCRE ranges
        dna_tokenizer: FastaInterval tokenizer (will be recreated in this function)
        interp_diff_dir: Directory containing .pt files
        output_dir: Output directory for results
        fasta_file: Path to fasta file
        context_length: Context length for tokenizer

    Returns:
        tuple: (celltype, output_path, summary_stats)
    """
    # Create local logger for this process
    logger = BaseLogger(name=f"DiffExpress_cCRE_{celltype}", level=logging.INFO)

    # Recreate tokenizer in this process (to avoid pickling issues)
    local_tokenizer = FastaInterval(
        fasta_file=os.path.abspath(fasta_file),
        context_length=context_length
    )

    # Get unique genes for this cell type
    genes = sorted(celltype_files['gene'].unique())

    # Initialize dataframe: rows=cCREs, columns=genes
    celltype_df = pd.DataFrame(
        index=ccre_df['Name'].values,
        columns=genes,
        dtype=float
    )

    # Process each file
    for _, row in celltype_files.iterrows():
        pt_file_path = os.path.join(interp_diff_dir, row['filename'])

        # Calculate cCRE attributions
        ccre_scores = calculate_ccre_attribution(
            pt_file_path=pt_file_path,
            chr_name=row['chr'],
            start=row['start'],
            end=row['end'],
            ccre_ranges=ccre_ranges,
            dna_tokenizer=local_tokenizer,
            logger=logger
        )

        # Update dataframe
        for ccre_name, score in ccre_scores.items():
            if ccre_name in celltype_df.index and row['gene'] in celltype_df.columns:
                if pd.isna(celltype_df.loc[ccre_name, row['gene']]):
                    celltype_df.loc[ccre_name, row['gene']] = score
                else:
                    celltype_df.loc[ccre_name, row['gene']] += score

    # Save dataframe
    output_path = os.path.join(output_dir, f"{celltype}_cCRE_gene_attribution.csv")
    celltype_df.to_csv(output_path)

    # Calculate summary statistics
    non_zero = (celltype_df != 0).sum().sum()
    total = celltype_df.shape[0] * celltype_df.shape[1]

    return celltype, output_path, (non_zero, total)


def main():
    # Setup
    logger = BaseLogger(name="DiffExpress_cCRE", level=logging.INFO)

    # Paths
    ccre_bed_path = "Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed"
    interp_diff_dir = "Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/raw_data/interp_diff"
    output_dir = "Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/DiffExpress_cCRE"

    os.makedirs(output_dir, exist_ok=True)

    # Load configuration to get fasta file path
    config_path = "logs/full_finetune_original_loss_celltype_head_dim8_linear/overall_setting.yaml"
    myconfig = load_config(config_name=config_path, skip_validation=True)

    # Load cCRE bed file
    logger.info(f"Loading cCRE bed file from {ccre_bed_path}...")
    ccre_df = pd.read_csv(ccre_bed_path, sep='\t', header=None,
                          names=['Chromosome', 'Start', 'End', 'Name', 'Score'])

    # Convert to PyRanges for efficient overlap operations
    ccre_ranges = pr.PyRanges(ccre_df[['Chromosome', 'Start', 'End', 'Name']])
    logger.info(f"Loaded {len(ccre_df)} cCREs")

    # Get all .pt files
    logger.info(f"Scanning files in {interp_diff_dir}...")
    pt_files = [f for f in os.listdir(interp_diff_dir) if f.endswith('.pt')]
    logger.info(f"Found {len(pt_files)} .pt files")

    # Parse all filenames and organize by cell type
    logger.info("Parsing filenames...")
    file_metadata = []
    for filename in pt_files:
        try:
            metadata = parse_filename(filename)
            file_metadata.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to parse filename {filename}: {str(e)}")

    metadata_df = pd.DataFrame(file_metadata)
    cell_types = sorted(metadata_df['celltype'].unique())
    logger.info(f"Found {len(cell_types)} cell types: {', '.join(cell_types)}")

    # Prepare arguments for parallel processing
    celltype_args = []
    for celltype in cell_types:
        celltype_files = metadata_df[metadata_df['celltype'] == celltype].copy()
        logger.info(f"Cell type {celltype}: {len(celltype_files)} files, {len(celltype_files['gene'].unique())} genes")

        celltype_args.append((
            celltype,
            celltype_files,
            ccre_df,
            ccre_ranges,
            None,  # dna_tokenizer will be recreated in each process
            interp_diff_dir,
            output_dir,
            myconfig.data.refer_genom,
            myconfig.data.context_length
        ))

    # Process cell types in parallel using joblib
    logger.info(f"\nProcessing {len(cell_types)} cell types in parallel...")

    results = Parallel(n_jobs=36, verbose=10)(
        delayed(process_celltype)(*args) for args in celltype_args
    )

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Processing complete!")
    logger.info("="*80)
    for celltype, output_path, (non_zero, total) in results:
        logger.info(f"{celltype}: {non_zero}/{total} ({100*non_zero/total:.2f}%) non-zero entries")
        logger.info(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()

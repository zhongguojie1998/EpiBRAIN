AlphaGenome eQTL Predictions Feather File
==========================================

File: eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather

Download URL:
  https://storage.googleapis.com/alphagenome/evals/eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather

Source:
  Google DeepMind AlphaGenome research repository
  https://github.com/google-deepmind/alphagenome_research

  Specifically referenced in the notebook:
  colabs/variant_eval_examples.ipynb

  The notebook defines:
    PREDS_PATH = 'https://storage.googleapis.com/alphagenome/evals'

  And constructs file paths as:
    os.path.join(PREDS_PATH, eval_name + '_predictions' + '.feather')

  where eval_name = 'eqtl_variant_catalogue_causality_gene_balanced_human'

Context:
  This file contains AlphaGenome model predictions for a balanced eQTL variant
  benchmark. The benchmark uses variants from the EMBL-EBI eQTL Catalogue
  (GTEx v8 reprocessed), selecting likely-causal (PIP > 0.9) and non-causal
  (PIP < 0.1) eQTL variants with balanced representation across tissue types
  and TSS distances (~23,058 variants total). The benchmark is used to evaluate
  variant effect prediction for causality (AUROC metric).

  The eQTL Catalogue data is licensed under CC-BY-4.0:
    "A compendium of uniformly processed human gene expression and splicing
     quantitative trait loci."

Related eval benchmarks hosted at the same GCS bucket:
  - eqtl_variant_borzoi_sign_human_predictions.feather
  - eqtl_variant_borzoi_coefficient_human_predictions.feather
  - sqtl_variant_causality_gene_human_predictions.feather
  - paqtl_variant_causality_human_predictions.feather
  - caqtl_african_variant_causality_human_predictions.feather
  - caqtl_european_variant_causality_human_predictions.feather
  - dsqtl_yoruba_variant_causality_human_predictions.feather
  - clinvar_noncoding_predictions.feather
  - clinvar_splice_site_region_predictions.feather
  - clinvar_missense_predictions.feather
  - mfass_splicing_predictions.feather
  - enhancer_gene_linking_e2g_predictions.feather
  (and more — see variant_eval_examples.ipynb for full list)

Publication:
  Avsec et al., "Advancing regulatory variant effect prediction with
  AlphaGenome", Nature (2025).
  https://www.nature.com/articles/s41586-025-10014-0

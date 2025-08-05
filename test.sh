python -u Analysis/03_variant_effect_screen/compute.py \
  --hdf5_file ./Data/source/GWAS_Var/res_file_250719_atac_rna_ft.h5 \
  --chunk_indices /home/dl3738/work/BICAN/test/parkinson/chunk_1_indices.npy \
  --model_path ./Chk/250719_atac_rna_ft/packaged.pkl \
  --device cuda:4 \
  --batch_size 1 \
  --save_interval 20000 \
  --precision float32 \
  --use_head human \
  --score_names raw_diff --score_names log_square
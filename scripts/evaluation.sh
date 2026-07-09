clear


# -------------------------------- iHarmony4 -------------------------------- # 
# CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py \
# --dataroot /path/to/iHarmony4/ \
# --result_root  results/iHarmony4/ \
# --evaluation_type our \
# --dataset_name ALL


CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation_csv.py \
--dataroot /path/to/iHarmony4/ \
--result_root  results/iHarmony4/ \
--evaluation_type our \
--dataset_name ALL 


# -------------------------------- ccHarmony -------------------------------- # 
# CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py \
# --dataroot /path/to/ccHarmony/ \
# --result_root  results/ccHarmony/ \
# --evaluation_type our \
# --dataset_name ALL


# CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation_csv.py \
# --dataroot /path/to/datasets/ \
# --result_root  results/ccHarmony/ \
# --evaluation_type our \
# --dataset_name ccHarmony 















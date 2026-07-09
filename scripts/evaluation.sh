clear


# -------------------------------- iHarmony4 -------------------------------- # 
# CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py \
# --dataroot /data0/zzr/datasets/iHarmony4/ \
# --result_root  results/iHarmony4/ \
# --evaluation_type our \
# --dataset_name ALL


CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation_csv.py \
--dataroot /data0/zzr/datasets/iHarmony4/ \
--result_root  results/iHarmony4/ \
--evaluation_type our \
--dataset_name ALL 


# -------------------------------- ccHarmony -------------------------------- # 
# CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py \
# --dataroot /data0/zzr/datasets/ccHarmony/ \
# --result_root  results/ccHarmony/ \
# --evaluation_type our \
# --dataset_name ALL


# CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation_csv.py \
# --dataroot /data0/zzr/datasets/ \
# --result_root  results/ccHarmony/ \
# --evaluation_type our \
# --dataset_name ccHarmony 















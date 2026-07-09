# # iHarmony4
# CUDA_VISIBLE_DEVICES=0 python Harmonization/test.py \
#     --dataset_root /path/to/iHarmony4 \
#     --weights checkpoints/wavssnet.pth \
#     --yaml_file Harmonization/Options/wavssnet.yml \
#     --datasets HAdobe5k HCOCO HFlickr Hday2night \
#     --result_dir results/iHarmony4/ \
#     --image_size 256



# ccHarmony
CUDA_VISIBLE_DEVICES=0 python Harmonization/test.py \
    --dataset_root /path/to/datasets \
    --weights checkpoints/wavssnet_ccharmony.pth \
    --yaml_file Harmonization/Options/wavssnet.yml \
    --datasets ccHarmony \
    --result_dir results/ccHarmony \
    --image_size 256



















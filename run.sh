export PYTHONPATH="/home/bd261576/PycharmProjects/swav":$PYTHONPATH

DIR="/neurospin/psy_sbox/bd261576/checkpoints/self_supervision/SwAV/ImageNet100/b256"
DATA="/neurospin/psy_sbox/bd261576/vision_datasets/ImageNet"

python3 -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
--dump_path $DIR \
--data_path $DATA \
--checkpoint_freq 100 \
--epochs 400 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 64 \
--size_crops 224 \
--nmb_crops 2 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--use_fp16 false \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--workers 30 \
--epoch_queue_starts 15 &> swav_imagenet100.log

python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py \
--dump_path $DIR \
--data_path $DATA \
--workers 30 \
--pretrained $DIR/ckp-399.pth.tar &> test_swav_imagenet100.log
PREFIX=w2v_unsup_gan_xp

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=w2vu2
# CONFIG_DIR=$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan
CONFIG_DIR=$EXP/w2vu2/config/gan
TASK_DATA=$DATA/LibriSpeech/features/wav2vec_vox

# Unpaired text input
TEXT_DATA=$DATA/variety-text-corpus/ImageCaptions/text/phones/
KENLM_PATH=$DATA/variety-text-corpus/ImageCaptions/text/phones/lm.phones.filtered.04.bin

export WANDB_TAGS=lr_sweep,debug.

lr_sweep=( [0.00005] [0.00009] )

for lr in "${lr_sweep[@]}"
do
  PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
      -m --config-dir $CONFIG_DIR \
      --config-name $CONFIG_NAME \
      task.data=${TASK_DATA} \
      task.text_data=${TEXT_DATA} \
      task.kenlm_path=${KENLM_PATH} \
      common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
      dataset.batch_size=2 \
      dataset.validate_interval=5 \
      checkpoint.save_interval=5 \
      checkpoint.save_interval_updates=5 \
      optimization.max_update=10 \
      optimizer.groups.generator.optimizer.lr=$lr
done

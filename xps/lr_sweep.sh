PREFIX=w2v_unsup_gan_xp

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=w2vu2
# CONFIG_DIR=$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan
CONFIG_DIR=$EXP/w2vu2/config/gan
TASK_DATA=$DATA/LibriSpeech-10hr-rVAD/features/wav2vec_vox

# Unpaired text input
TEXT_DATA=$DATA/variety-text-corpus/LibriLM/text/phones/
KENLM_PATH=$DATA/variety-text-corpus/LibriLM/text/phones/lm.phones.filtered.04.bin

export WANDB_TAGS=lr_sweep

lr_sweep_g=( [0.00001] [0.00002] [0.00003] [0.00004] [0.00005] )
lr_sweep_d=( [0.0003] [0.0002] [0.0001] )

for lr_g in "${lr_sweep_g[@]}"
do
for lr_d in "${lr_sweep_d[@]}"
do
  PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
      -m --config-dir $CONFIG_DIR \
      --config-name $CONFIG_NAME \
      task.data=${TASK_DATA} \
      task.text_data=${TEXT_DATA} \
      task.kenlm_path=${KENLM_PATH} \
      common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
      optimization.max_update=500 \
      optimizer.groups.generator.optimizer.lr=$lr_g \
      optimizer.groups.discriminator.optimizer.lr=$lr_d \
      model.smoothness_weight=0 \
      model.mmi_weight=0 \
      model.code_penalty=0 \
done
done

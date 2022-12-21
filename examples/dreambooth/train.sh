export MODEL_NAME="/home/samuel_liard/models/v1-5-pruned-emaonly.ckpt"
export INSTANCE_DIR="/home/samuel_liard/diffusers/examples/gpeyronnet"
export CLASS_IMAGES_DIR="/home/samuel_liard/diffusers/examples/person"
export OUTPUT_DIR="/home/samuel_liard/diffusers/examples/dreambooth/output"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path $MODEL_NAME  \
  --instance_data_dir $INSTANCE_DIR \
  --output_dir $OUTPUT_DIR \
  --class_data_dir $CLASS_IMAGES_DIR \
  --with_prior_preservation \
  --class_prompt "a photo of a person" \
  --instance_prompt="a photo of sks person" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --max_train_steps 400
export GLUE_DIR=../data/glue_data/
export TASK_NAME=SST-2

python run_glue.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 1e-5 \
  --warmup_steps 1256 \
  --save_steps 1000 \
  --max_steps 20935 \
  --logging_steps 1000 \
  --eval_all_checkpoints \
  --output_dir ../models/roberta/$TASK_NAME/

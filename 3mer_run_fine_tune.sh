#!/bin/sh
#module load anaconda/2020.11
source activate tf15
export PYTHONUNBUFFERED=1
export DATA_DIR=data/3mer_tokenize_data/4mC/4mC_C.equisetifolia
export BERT_BASE_DIR=5mc_pre_train/5mc-3;
python my_run_classifier.py --task_name=mytask --do_train=true --do_eval=false --do_predict=true --do_lower_case=True --data_dir=$DATA_DIR --vocab_file=vocab/vocab_3kmer.txt --bert_config_file=config/bert_config_3.json --init_checkpoint=5mc_pre_train/5mc-3/model.ckpt-200000 --max_seq_length=128 --train_batch_size=16 --learning_rate=2e-5  --num_train_epochs=10.0 --output_dir=fine_tune/3mer/4mC/4mC_C.equisetifolia


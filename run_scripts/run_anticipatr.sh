#!/bin/bash

nohup python -m torch.distributed.run --nproc_per_node=6 --master_addr=localhost --master_port=12355 src/main.py --dataset ek --root data --num_verbs 125 --num_nouns 352 --num_actions 3806 --action_repr actionset --num_queries 900 --anticipation longfuture --label_type verb --pretrained_enc_layers 3 --pretrained_path  "pretraining_expts/checkpoints/try/checkpoint.pth" &
# python -m torch.distributed.launch --nproc_per_node=2 src/main.py --dataset ek --root data --num_verbs 125 --num_nouns 352 --num_actions 3806 --action_repr actionset --num_queries 900 --anticipation longfuture --label_type verb --pretrained_enc_layers 3 --pretrained_path  "pretraining_expts/checkpoints/try/checkpoint.pth" 

# python src/main.py --dataset ek --root data --num_verbs 125 --num_nouns 352 --num_actions 3806 --action_repr actionset --num_queries 900 --anticipation longfuture --label_type verb --pretrained_enc_layers 3 --pretrained_path  "pretraining_expts/checkpoints/try/checkpoint.pth"

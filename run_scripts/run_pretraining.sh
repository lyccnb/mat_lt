#!/bin/bash

python pretraining/main_pretraining.py --dataset ek --root data  --pretraining_task snippet_longfuture_anticipation --output_dir ./pretraining_expts/checkpoints/try --dataset ek --root ./data --pretraining_task snippet_longfuture_anticipation --output_dir ./pretraining_expts/checkpoints/try --num_verbs 125 --num_nouns 352 --num_actions 3806 --num_queries 900

#!/bin/bash

python run.py --run-id theor_step_size --model log_reg --method sgd_8192 --dataset ijcnn1 --lr 0.125 --epochs 20 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method svrg_8192 --dataset ijcnn1 --lr 0.125 --epochs 20 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method sarah_8192 --dataset ijcnn1 --lr 0.125 --epochs 20 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method scsg_low --dataset ijcnn1 --lr 0.125 --epochs 20 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method q-sarah_low --dataset ijcnn1 --lr 0.125 --epochs 20 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method e-sarah_low --dataset ijcnn1 --lr 0.125 --epochs 20 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5
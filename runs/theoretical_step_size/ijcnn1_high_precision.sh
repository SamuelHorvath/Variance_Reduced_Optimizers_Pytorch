#!/bin/bash

python run.py --run-id theor_step_size --model log_reg --method gd --dataset ijcnn1 --lr 1.615 --epochs 50 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method svrg --dataset ijcnn1 --lr 0.125 --epochs 50 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method sarah --dataset ijcnn1 --lr 0.125 --epochs 50 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method scsg_high --dataset ijcnn1 --lr 0.125 --epochs 50 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method q-sarah_high --dataset ijcnn1 --lr 0.125 --epochs 50 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5

python run.py --run-id theor_step_size --model log_reg --method e-sarah_high --dataset ijcnn1 --lr 0.125 --epochs 50 -b 2048 --metric loss --loss BCE --deterministic --num-workers 0 --track-grad-norm --nc-regularizer --nc-regularizer-value 2e-5
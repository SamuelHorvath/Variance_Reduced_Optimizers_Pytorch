#!/bin/bash

python run.py --run-id neural_nets --model fcn --method scsg_1024 --dataset mnist --lr 0.05 --epochs 21 -b 4096 --metric loss --loss CE --deterministic --num-workers 0

python run.py --run-id neural_nets --model fcn --method q-sarah_1024 --dataset mnist --lr 0.025 --epochs 21 -b 4096 --metric loss --loss CE --deterministic --num-workers 0

python run.py --run-id neural_nets --model fcn --method e-sarah_1024 --dataset mnist --lr 0.025 --epochs 21 -b 4096 --metric loss --loss CE --deterministic --num-workers 0

python run.py --run-id neural_nets --model fcn --method sgd_1024 --dataset mnist --lr 0.1 --epochs 21 -b 4096 --metric loss --loss CE --deterministic --num-workers 0

python run.py --run-id neural_nets --model fcn --method svrg_1024 --dataset mnist --lr 0.05 --epochs 21 -b 4096 --metric loss --loss CE --deterministic --num-workers 0

python run.py --run-id neural_nets --model fcn --method sarah_1024 --dataset mnist --lr 0.025 --epochs 21 -b 4096 --metric loss --loss CE --deterministic --num-workers 0

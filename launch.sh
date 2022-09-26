#!/bin/bash


python server_fed.py --world_size 2 &

python client_fed.py --world_size 2 --rank 1 --epoch 50 --lr 5e-5 &


wait
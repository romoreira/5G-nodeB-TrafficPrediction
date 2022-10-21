#!/bin/bash


python server.py --world_size 2 --client_id 1 &

python client.py --world_size 2 --rank 1 --epoch 30 --lr 0.001 --client_id 1 &
#python client.py --world_size 4 --rank 2 --epoch 30 --lr 0.001 --client_id 2 &
#python client.py --world_size 4 --rank 3 --epoch 30 --lr 0.001 --client_id 3 &


wait
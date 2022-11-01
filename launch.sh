#!/bin/bash


python3 server_fed.py --world_size 2 --client_id 1 --num_workers 0 --batch_size 30 &

python3 client_fed.py --world_size 2 --rank 1 --epoch 30 --num_workers 0 --lr 0.001 --batch_size 30 --client_id 1 &
#python3 client_fed.py --world_size 4 --rank 2 --epoch 10 --num_workers 0 --lr 0.01 --batch_size 30 --client_id 2 &
#python3 client_fed.py --world_size 4 --rank 3 --epoch 10 --num_workers 0 --lr 0.01 --batch_size 30 --client_id 3 &
wait
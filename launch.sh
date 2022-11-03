#!/bin/bash


python3 server_fed.py --world_size 3 --client_id 2 --num_workers 0 --batch_size 32 &

python3 client_fed.py --world_size 3 --rank 1 --epoch 30 --num_workers 0 --lr 0.001 --batch_size 32 --client_id 1 &
#python3 client_fed.py --world_size 3 --rank 2 --epoch 30 --num_workers 0 --lr 0.001 --batch_size 32 --client_id 2 &
python3 client_fed.py --world_size 3 --rank 2 --epoch 30 --num_workers 0 --lr 0.001 --batch_size 32 --client_id 3 &
wait
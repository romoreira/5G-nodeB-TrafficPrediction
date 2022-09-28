#!/bin/bash


python server_fed.py --world_size 4 --client_id 1 &

python client_fed.py --world_size 4 --rank 1 --epoch 2 --lr 5e-5 --client_id 1 &
python client_fed.py --world_size 4 --rank 2 --epoch 2 --lr 5e-5 --client_id 2 &
python client_fed.py --world_size 4 --rank 3 --epoch 2 --lr 5e-5 --client_id 3 &


wait
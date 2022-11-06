#!/bin/bash

python3 server.py &
python3 client.py --client_id 1 &
python3 client.py --client_id 2 &
wait
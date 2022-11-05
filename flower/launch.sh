#!/bin/bash

python3 server.py &
python3 client.py &
python3 client.py &

wait
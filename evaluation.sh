#!/usr/bin/env sh
set -e
EVAL_ITER=1000
cd util
python "evaluation.py" --iter=$EVAL_ITER
cd ..
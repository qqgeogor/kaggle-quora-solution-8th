#!/bin/sh
# mining pattern
python pattern_mining.py 1
python pattern_mining.py 0
# compute mi
python mi.py
# nn models for stacking
mkdir cv
mkdir cv/model
mkdir cv/log
mkdir cv/stack
THEANO_FLAGS="floatX=float32,device=cuda,optimizer=fast_run,lib.cnmem=1,nvcc.fastmath=True"  python jasonnet_drop_naive.py
THEANO_FLAGS="floatX=float32,device=cuda,optimizer=fast_run,lib.cnmem=1,nvcc.fastmath=True"  python jasonnet_drop_match.py
THEANO_FLAGS="floatX=float32,device=cuda,optimizer=fast_run,lib.cnmem=1,nvcc.fastmath=True"  python jasonnet_drop_diff.py
cd utils
python cv4stack.py
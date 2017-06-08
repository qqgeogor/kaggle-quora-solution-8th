#!/bin/sh
THEANO_FLAGS="floatX=float32,device=cuda,optimizer=fast_run,lib.cnmem=1,nvcc.fastmath=True"  python main.py --train_file data/train.txt.train.2  --goto_line 0  --batch_size 1000

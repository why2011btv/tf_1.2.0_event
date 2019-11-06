mkdir results_9
CUDA_VISIBLE_DEVICES=0 python run_bilstm.py 1 1 128 4 concat ./results_9/typed_bilstm_concat_128_4l.txt
CUDA_VISIBLE_DEVICES=0 python run_bilstm.py 1 1 128 4 subtract ./results_9/typed_bilstm_subtract_128_4l.txt
CUDA_VISIBLE_DEVICES=1 python run_bilstm.py 1 1 128 5 concat ./results_9/typed_bilstm_concat_128_5l.txt
CUDA_VISIBLE_DEVICES=1 python run_bilstm.py 1 1 256 4 concat ./results_9/typed_bilstm_concat_256_5l.txt

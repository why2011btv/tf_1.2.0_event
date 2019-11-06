#CUDA_VISIBLE_DEVICES=4 python run_bilstm.py 1 0 128 4 concat ./results/bilstm_concat_128_4l.txt
#CUDA_VISIBLE_DEVICES=4 python run_bilstm.py 1 0 128 4 subtract ./results/bilstm_subtract_128_4l.txt
#CUDA_VISIBLE_DEVICES=4 python run_bilstm.py 1 0 128 3 concat ./results/bilstm_concat_128_3l.txt
#CUDA_VISIBLE_DEVICES=4 python run_bilstm.py 1 0 128 2 concat ./results/bilstm_concat_128_2l.txt
#CUDA_VISIBLE_DEVICES=4 python run_bilstm.py 1 0 64 2 concat ./results/bilstm_concat_64_2l.txt
#CUDA_VISIBLE_DEVICES=4 python run_bilstm.py 1 0 256 3 concat ./results/bilstm_concat_256_3l.txt
#CUDA_VISIBLE_DEVICES=4 python run_bilstm.py 1 0 256 3 subtract ./results/bilstm_subtract_256_3l.txt



#CUDA_VISIBLE_DEVICES=6 python run_bitcn.py 1 0 128 4 concat ./results/bitcn_concat_128_4l.txt
#CUDA_VISIBLE_DEVICES=6 python run_bitcn.py 1 0 128 4 subtract ./results/bitcn_subtract_128_4l.txt
#CUDA_VISIBLE_DEVICES=6 python run_bitcn.py 1 0 128 3 subtract ./results/bitcn_subtract_128_3l.txt
#CUDA_VISIBLE_DEVICES=6 python run_bitcn.py 1 0 128 3 concat ./results/bitcn_concat_128_3l.txt
#CUDA_VISIBLE_DEVICES=6 python run_bitcn.py 1 0 256 3 subtract ./results/bitcn_subtract_256_3l.txt
#CUDA_VISIBLE_DEVICES=6 python run_bitcn.py 1 0 256 3 concat ./results/bitcn_concat_256_3l.txt



# 9/13
mkdir results_9
CUDA_VISIBLE_DEVICES=0 python run_bilstm.py 1 1 128 4 concat ./results_9/typed_bilstm_concat_128_4l.txt
#CUDA_VISIBLE_DEVICES=0 python run_bilstm.py 1 1 128 4 subtract ./results_9/typed_bilstm_subtract_128_4l.txt
#CUDA_VISIBLE_DEVICES=1 python run_bilstm.py 1 1 128 5 concat ./results_9/typed_bilstm_concat_128_5l.txt
#CUDA_VISIBLE_DEVICES=1 python run_bilstm.py 1 1 256 4 concat ./results_9/typed_bilstm_concat_256_5l.txt


#mkdir results_9v
#CUDA_VISIBLE_DEVICES=5 python run_bilstm_fverb.py 1 0 128 4 concat ./results_9v/fverb_bilstm_concat_128_4l.txt
#CUDA_VISIBLE_DEVICES=5 python run_bilstm_fverb.py 1 0 128 4 subtract ./results_9v/fverb_bilstm_subtract_128_4l.txt
#CUDA_VISIBLE_DEVICES=5 python run_bilstm_fverb.py 1 0 128 5 concat ./results_9v/fverb_bilstm_concat_128_5l.txt
#CUDA_VISIBLE_DEVICES=5 python run_bilstm_fverb.py 1 0 256 4 concat ./results_9v/fverb_bilstm_concat_256_5l.txt

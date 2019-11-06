# event_abstraction
cd ./baselines
CUDA_VISIBLE_DEVICES=4 nohup python run_bilstm.py 1 1 128 6 concat ./results_9/typed_bilstm_concat_128_6l.txt > typed_bilstm_concat_128_6l.out 2>&1 &

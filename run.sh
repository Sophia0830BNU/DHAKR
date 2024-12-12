nohup python3 -u main.py  --dataset MUTAG --lr 0.004 --device 3 --epochs 500 --feature_hid 50 --layers 3 --num_mlp_layer 3 --method WL --iteration_num 1 --loss_alpha 0.01 >>MUTAG_WL.out&

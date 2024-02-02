#train
python3 tools/train.py --n_way 3 --k_shot 5 --k_query 20 --epochs 40 --path_train "data/dic_1/train/" --config_path "configs/maml.yml" --weights_path "weights/maml.pt" --using_pretrain "False" --save_weights "True"
#test
python3 tools/test.py --n_way 3 --k_shot 5 --k_query 20 --path_test "data/dic_1/test/" --config_path "configs/maml.yml" --weights_path "weights/maml.pt"
python3 tools/train.py --n_way 2 --n_shot 10 --n_query 20 --tasks 1000 --batch_size 16 --path_train "data/train/" --config_path "configs/maml.yml" --weights_path "weights/maml.pt" --using_pretrain False --save_weights True
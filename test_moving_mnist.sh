# for train
python main.py --dataset mmnist --pred_len 10 --in_shape "(10, 1, 64, 64)" --lr 1e-3 --sched cosine --epochs 2000 --batch_size 16 --test_batch_size 16 --save_path moving_mnist --embed_dim 128 --patch_size 2 --N 8 --save_result False
# for test
python test.py --dataset mmnist --pred_len 10 --in_shape "(10, 1, 64, 64)" --lr 1e-3 --sched cosine --epochs 2000 --batch_size 16 --test_batch_size 16 --save_path moving_mnist --embed_dim 128 --patch_size 2 --N 8 --pr_path pred_mmnist.npy --gt_path gt_mmnist.npy
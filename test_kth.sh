# for train
python main.py --dataset kth20 --pred_len 20 --in_shape "(10, 1, 128, 128)" --lr 1e-3 --epochs 100 --batch_size 4 --test_batch_size 4 --save_path kth --embed_dim 128 --patch_size 2 --N 6 --droppath 0.2 --save_result False
# for test
python test.py --dataset kth20 --pred_len 20 --in_shape "(10, 1, 128, 128)" --lr 1e-3 --epochs 100 --batch_size 4 --test_batch_size 4 --save_path kth --embed_dim 128 --patch_size 2 --N 6 --droppath 0.2 --pr_path pred_kth.npy --gt_path gt_kth.npy

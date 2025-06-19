# for train
python main.py --dataset taxibj --pred_len 4 --in_shape "(4, 2, 32, 32)" --lr 1e-3 --sched cosine --epochs 100 --batch_size 16 --test_batch_size 16 --save_path taxibj --device 0 --embed_dim 128 --patch_size 4 --N 6 --save_result False
# for test
python test.py --dataset taxibj --pred_len 4 --in_shape "(4, 2, 32, 32)" --lr 1e-3 --sched cosine --epochs 100 --batch_size 16 --test_batch_size 16 --save_path taxibj --device 0 --embed_dim 128 --patch_size 4 --N 6 --pr_path pred_taxibj.npy --gt_path gt_taxibj.npy

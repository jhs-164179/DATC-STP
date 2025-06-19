import time
import warnings
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, flop_count_table
from utils import *
from metrics import *
from model import DATCSTP
from dataset import MovingMNIST, TaxibjDataset, load_kth


def make_mmnist(args):
    train_set = MovingMNIST(root='./dataset/moving_mnist', is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
    test_set = MovingMNIST(root='./dataset/moving_mnist', is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    xx, yy = next(iter(train_loader))
    print(xx.shape, yy.shape)
    return train_loader, test_loader


def make_taxibj(args):
    dataset = np.load('./dataset/taxibj/dataset.npz')
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
    train_set = TaxibjDataset(X=X_train, Y=Y_train)
    test_set = TaxibjDataset(X=X_test, Y=Y_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    xx, yy = next(iter(train_loader))
    print(xx.shape, yy.shape)
    return train_loader, test_loader


def make_kth20(args):
    train_set, test_set = load_kth(
        batch_size=args.batch_size, val_batch_size=args.test_batch_size,
        data_root='./dataset'
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    xx, yy = next(iter(train_loader))
    print(xx.shape, yy.shape)
    return train_loader, test_loader


def calculate_model_size(model, args):
    flops = FlopCountAnalysis(model, torch.randn(1, *args.in_shape))
    gflops = flops.total() / 1e9
    flop_table = flop_count_table(flops)
    return gflops, flop_table


def test_save(model, args):
    gflops, flop_table = calculate_model_size(model, args)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    ckpt = torch.load(os.path.join('./logs', args.save_path, 'best_model.pth'), map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    
    if args.dataset == 'mmnist':
        _, test_loader = make_mmnist(args)
    elif args.dataset == 'taxibj':
        _, test_loader = make_taxibj(args)
    elif args.dataset == 'kth20':
        _ ,test_loader = make_kth20(args)
    else:
        raise NotImplementedError

    save_path = f'./logs/{args.save_path}'
    check_path(save_path)
    with open(os.path.join(save_path, f'inference.txt'), 'w') as log_file:
        print_f(model, file=log_file)
        print_f(flop_table, file=log_file)
        print_f(f'model params: {count_parameters(model)}', file=log_file)
        print_f(f'model gflops: {gflops}', file=log_file)

        # evaluate with various metrics
        model.eval()
        test_samples = 0
        pred_save = []
        gt_save = []
        t12 = time.time()
        with torch.no_grad():
            total_mse, total_mae, total_psnr, total_ssim = 0, 0, 0, 0
            print_f('===Final Evaluation===', file=log_file)
            for X, y in tqdm(test_loader):
                X = X.to(device)
                if args.dataset == 'kth20':
                    preds = []
                    d = args.pred_len // args.in_shape[0]
                    m = args.pred_len % args.in_shape[0]
                    cur_seq = X.clone()
                    for _ in range(d):
                        cur_seq = model(cur_seq)
                        preds.append(cur_seq)
                    if m != 0:
                        cur_seq = model(cur_seq)
                        preds.append(cur_seq[:, :m])
                    preds = torch.cat(preds, dim=1)
                else:
                    preds = model(X)                
                
                preds = preds.detach().cpu().numpy()
                true = y.numpy()

                total_mse += MSE(preds, true) * X.size(0)
                total_mae += MAE(preds, true) * X.size(0)

                preds = np.maximum(preds, 0)
                preds = np.minimum(preds, 1)

                total_psnr += cal_psnr(preds, true) * X.size(0)
                total_ssim += cal_ssim(preds, true) * X.size(0)

                test_samples += X.size(0)

                if args.save_result:    
                    pred_save.append(preds)
                    gt_save.append(preds)

            total_mse /= test_samples
            total_mae /= test_samples
            total_psnr /= test_samples
            total_ssim /= test_samples

            if args.save_result:
                ppred_save = np.concatenate(pred_save, axis=0)
                ggt_save = np.concatenate(gt_save, axis=0)
                print(f'Preds: {ppred_save.shape}')
                print(f'Ground Truth: {ggt_save.shape}')
                np.save(os.path.join(save_path, args.pr_path), ppred_save)
                np.save(os.path.join(save_path, args.gt_path), ggt_save)
            print_f(f'Inference Done | Time: {time.time() - t12:.4f}', file=log_file)    
            print_f(f'MSE: {total_mse:.6f} | MAE: {total_mae:.6f} | PSNR: {total_psnr:.6f} | SSIM: {total_ssim:.6f} | Time: {time.time() - t12:.4f}', file=log_file)    

def main():
    seed_everything()
    args = makeparser()
    model = DATCSTP(
        args.in_shape[1], embed_dim=args.embed_dim, patch_size=args.patch_size, 
        N=args.N, kernel_size=args.S_kernel, T=args.in_shape[0], ratio=args.ratio, drop_path=args.droppath
    )
    test_save(model, args)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
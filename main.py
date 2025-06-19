import time
import torch
import warnings
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from dataset import MovingMNIST, TaxibjDataset, load_kth
from model import DATCSTP
from metrics import *
from utils import *


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



def train_test(model, args):
    gflops, flop_table = calculate_model_size(model, args)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # scheduler
    if args.sched == 'onecycle':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=10
        )
    else:
        scheduler = CosineLRScheduler(
            optimizer=optimizer, t_initial=args.epochs, lr_min=1e-06, warmup_lr_init=1e-05, warmup_t=5
        )

    criterion1, criterion2 = nn.MSELoss(), nn.L1Loss()
    if args.dataset == 'mmnist':
        train_loader, test_loader = make_mmnist(args)
    elif args.dataset == 'taxibj':
        train_loader, test_loader = make_taxibj(args)
    elif args.dataset == 'kth20':
        train_loader ,test_loader = make_kth20(args)
    else:
        raise NotImplementedError

    best_loss = np.inf
    best_epoch = 0
    save_path = f'./logs/{args.save_path}'
    check_path(save_path)
    with open(os.path.join(save_path, f'{args.epochs}epochs.txt'), 'w') as log_file:
        print_f(model, file=log_file)
        print_f(flop_table, file=log_file)
        print_f(f'model params: {count_parameters(model)}', file=log_file)
        print_f(f'model gflops: {gflops}', file=log_file)
        for epoch in range(args.epochs):
            # train
            model.train()
            t0 = time.time()
            loss_train = 0.0
            train_samples = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                
                # for kth only (input_len != pred_len)
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
                # for mmnist or taxibj
                else:
                    preds = model(X)
                    
                if args.loss == 'l1l2':
                    loss = 10*criterion1(preds, y) + criterion2(preds, y)
                else:
                    loss = criterion1(preds, y)
                loss.backward()
                optimizer.step()
                loss_train += loss.item() * X.size(0)
                train_samples += X.size(0)
            loss_train = loss_train / train_samples
            print_f(f'Epoch {epoch + 1} | Loss : {loss_train:.6f} | Time : {time.time() - t0:.4f}', file=log_file)

            # test
            model.eval()
            t1 = time.time()
            test_samples = 0
            with torch.no_grad():
                loss_test = 0.0
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    
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
                    
                    loss = criterion1(preds, y)
                    loss_test += loss.item() * X.size(0)
                    test_samples += X.size(0)
                loss_test = loss_test / test_samples
                if args.sched == 'onecycle':
                    scheduler.step(epoch=epoch)
                else:
                    scheduler.step(epoch=epoch, metric=loss_test)                
                print_f(f'Epoch {epoch + 1} | Test Loss : {loss_test:.6f} | Time : {time.time() - t1:.4f}', file=log_file)

                # save best model
                if loss_test < best_loss:
                    best_loss = loss_test
                    best_epoch = epoch + 1
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(save_path, 'best_model.pth'))
                    print_f(f'Best model saved with loss {best_loss:.6f} at epoch {epoch + 1}', file=log_file)
        # save last model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_path, 'last_model.pth'))

        # evaluate with various metrics
        checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test_samples = 0
        with torch.no_grad():
            total_mse, total_mae, total_psnr, total_ssim = 0, 0, 0, 0
            print_f('===Final Evaluation===', file=log_file)
            for X, y in test_loader:
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
            total_mse /= test_samples
            total_mae /= test_samples
            total_psnr /= test_samples
            total_ssim /= test_samples
            print_f(f'Best Epoch {best_epoch} | MSE : {total_mse:.6f} | MAE : {total_mae:.6f} | PSNR : {total_psnr:.6f} | SSIM : {total_ssim:.6f}', file=log_file)    
    

def main():
    seed_everything()
    args = makeparser()
    model = DATCSTP(
        args.in_shape[1], embed_dim=args.embed_dim, patch_size=args.patch_size, 
        N=args.N, kernel_size=args.S_kernel, T=args.in_shape[0], ratio=args.ratio, drop_path=args.droppath
    )
    train_test(model, args)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
import torch
from network import UCMDC
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
from loss import Loss
from load_data import load_data
import os
import load_data as loader
from datasets import Data_Sampler, DatasetLoad

################## main ##################
my_data_dic = loader.ALL_data

for i_d in my_data_dic:
    data_para = my_data_dic[i_d]
    print(data_para)
    txt_file = "./result/" + data_para[1] + ".txt"
    f = open(txt_file, 'a+')
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--temperature_f", default=0.5, type=float)#0.5
    parser.add_argument("--para_lambda", default=1,type=float)
    parser.add_argument('--dataset', default=data_para)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--learning_rate", default=1e-4)#1e-4
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--rec_epochs", default=200)#200
    parser.add_argument("--fine_tune_epochs", default=200)#200
    parser.add_argument("--low_feature_dim", default=512)
    parser.add_argument("--high_feature_dim", default=128)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X,Y = load_data(args.dataset)
    dataset = data_para[1]
    dims = data_para['n_input']
    view = data_para['V']
    data_size = data_para['N']
    class_num = data_para['K']

    train_dataset = DatasetLoad(X, Y)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    def pre_train(epoch):
        tot_loss = 0.
        mse = torch.nn.MSELoss()
        for batch_idx, (xs, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = torch.squeeze(xs[v]).to(device)
            optimizer.zero_grad()
            xrs, _, _ = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(mse(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    def fine_tune(epoch):
        tot_loss = 0.
        mes = torch.nn.MSELoss()
        for batch_idx, (xs, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = torch.squeeze(xs[v]).to(device)
            optimizer.zero_grad()
            xrs, _, hs = model(xs)
            commonz, S = model.GCFAgg(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(args.para_lambda* criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S))
                loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    model = UCMDC(view, dims, args.low_feature_dim, args.high_feature_dim, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
    epoch = 1
    while epoch <= args.rec_epochs:
        pre_train(epoch)
        epoch += 1
    acc_best=0
    stop_patience=50
    epoch_save = 0
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'models/model_{dataset}.pth'
    while epoch <= args.rec_epochs + args.fine_tune_epochs:
        fine_tune(epoch)
        if (epoch + 1) % 1 == 0:
            acc, nmi, pur, ari = valid(model, device, X, Y, view, class_num, bz=args.batch_size)
            if acc > acc_best:
                acc_best,nmi_bset, pur_best, ari_best,epoch_save = acc, nmi,pur,ari,epoch
                torch.save(model.state_dict(), model_path)
                acc_best = float(np.round(acc_best, 4))
                nmi_bset = float(np.round(nmi_bset, 4))
                pur_best = float(np.round(pur_best, 4))
                ari_best = float(np.round(ari_best, 4))
        if epoch - epoch_save >= stop_patience:
            print(f'Stopping early at epoch {epoch + 1}. Best ARI: {acc_best:.4f} at epoch {epoch_save + 1}')
            break
        print(acc_best)
        epoch += 1



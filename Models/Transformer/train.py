import numpy as np
import time
import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from common.util import auc_plotter, show_prog, save_prog, get_aucs
from common.dataloader import Dataset, collate_fn, make_train_loader, get_train_weights, get_valid_weights
from model import RT


# before IS
train_path = '../../Preprocessing/data/processed_data/train_tensors/'
valid_path = '../../Preprocessing/data/processed_data/valid_tensors/'
save_path = '../../local_data/models/Transformer/clean/'
#for transfer
#train_path = '../../local_data/srtr_data/multi_label/UHN_transfer/n_train_tensors/'
#valid_path = '../../local_data/srtr_data/multi_label/UHN_transfer/n_valid_tensors/'
#save_path = '../../local_data/models/transfer/Transformer/'

######## __GENERAL__ ########
parser = argparse.ArgumentParser(description='training control')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--epochs', action='store', default=40, type=int,
                    help='num epochs')
parser.add_argument('--batch', action='store', default=512, type=int,
                    help='batch size')
parser.add_argument('--nosave', action='store_true',
                    help='do not save flag')
parser.add_argument('--prog', action='store_true',
                    help='show progress')
parser.add_argument('--search', action='store_true',
                    help='search output formatting')
parser.add_argument('--cv', action='store', default=1, type=int,
                    help='Cross-Val fold')

######## __RTransformer__ ########
parser.add_argument('--dim', action='store', default=32, type=int,
                    help='hidden state dimension')
parser.add_argument('--heads', action='store', default=2, type=int,
                    help='number of attention heads')
parser.add_argument('--ksize', action='store', default=3, type=int,
                    help='receptive field of RNN')
parser.add_argument('--rnn', action='store', default=1, type=int,
                    help='number of RNN layers')
parser.add_argument('--levels', action='store', default=2, type=int,
                    help='number of attention layers')
parser.add_argument('--drop', action='store', default=0.1, type=float,
                    help='droprate')
parser.add_argument('--l2', action='store', default=0.01, type=float,
                    help='l2 regularization penalty')

######## __OPTIM__ ########
parser.add_argument('--lr', action='store', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('--b1', action='store', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--b2', action='store', default=0.999, type=float,
                    help='momentum')
args = parser.parse_args()

# train_path = '../../local_data/srtr_data/immuno/CV' + str(args.cv) + '/n_train_tensors/'
# valid_path = '../../local_data/srtr_data/immuno/CV' + str(args.cv) + '/n_valid_tensors/'
# save_path = '../../local_data/models/Transformer/CV' + str(args.cv) + '/IS/'

######## __GPU_SETUP__ ########
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    args.device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

def train(weights):

    running_loss  = 0
    correct = np.zeros((10))
    pos = np.zeros((10))
    total = 0
    predictions = [np.array([])] * 10 
    actual = [np.array([])] * 10
    t_neg_weights, t_class_weights = weights
    t_neg_weights, t_class_weights = t_neg_weights.to(args.device), t_class_weights.to(args.device)

    for batch, labels, seq_len in train_loader:
        # pass to GPU if available
        batch, labels = batch.to(args.device), labels.to(args.device)

        # run network
        optimizer.zero_grad()
        outputs = model(batch)

        multiplier = ( ((labels==0).double() * t_neg_weights) + (labels==1).double() ) * t_class_weights
        mask = (multiplier > 0) * (labels <= 1) * (labels >= 0)

        loss = torch.mean( criterion(outputs.masked_select(mask), labels.masked_select(mask)) * multiplier.masked_select(mask) )

        # adjust weights and record loss
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data.numpy()

        # train accuracy
        for i in range(labels.shape[0]):
            targets = labels.data[i][:int(seq_len[i])].cpu().numpy()
            prob = outputs.data[i][:int(seq_len[i])].cpu().numpy()

            prediction = np.zeros(targets.shape)
            # prediction[np.arange(prediction.shape[0]), np.argmax(prob[:, :5], axis=1)] = 1
            # prediction[np.arange(prediction.shape[0]), np.argmax(prob[:, 5:], axis=1)+5] = 1
            prediction = prob > 0.5
            match = (targets == prediction)


            # print('prob', prob)
            # print('targets', targets)
            # print('prediction', prediction)
            # print('match', match)

            pos += np.sum(prediction,axis=0).astype(int)
            correct += np.sum(match,axis=0).astype(int)
            total += targets.shape[0]

            # for AUC
            for j in range(10):
                actual[j] = np.concatenate((actual[j], labels[i][:int(seq_len[i]),j].view(-1).cpu().numpy()))
                predictions[j] = np.concatenate((predictions[j], outputs[i][:int(seq_len[i]),j].detach().view(-1).cpu().numpy()))

    train_losses[epoch] = running_loss/len(train_loader)
    train_acc[epoch] = correct/total
    train_freq[epoch] = pos / sum(pos)
    train_auc[epoch] = get_aucs(actual, predictions)
    
def valid(weights):
    
    running_loss  = 0
    correct = np.zeros(10)
    pos = np.zeros(10)
    total = 0
    predictions = [np.array([])] * 10
    actual = [np.array([])] * 10
    v_neg_weights, v_class_weights = weights
    v_neg_weights, v_class_weights = v_neg_weights.to(args.device), v_class_weights.to(args.device)

    with torch.no_grad():
        for batch, labels, seq_len in val_loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)

            # run network
            optimizer.zero_grad()
            outputs = model(batch)

            multiplier = ( ((labels==0).double() * v_neg_weights) + (labels==1).double() ) * v_class_weights
            mask = (multiplier > 0) * (labels <= 1) * (labels >= 0)

            loss = torch.mean( criterion(outputs.masked_select(mask), labels.masked_select(mask)) * multiplier.masked_select(mask) )

            running_loss += loss.cpu().data.numpy()
            
            # Validation accuracy
            for i in range(labels.shape[0]):
                targets = labels.data[i][:int(seq_len[i])].cpu().numpy()
                prob = outputs.data[i][:int(seq_len[i])].cpu().numpy()

                prediction = np.zeros(targets.shape)
                # prediction[np.arange(prediction.shape[0]), np.argmax(prob, axis=1)] = 1
                # prediction[np.arange(prediction.shape[0]), np.argmax(prob[:, :5], axis=1)] = 1
                # prediction[np.arange(prediction.shape[0]), np.argmax(prob[:, 5:], axis=1) + 5] = 1
                prediction = prob > 0.5
                match = (targets == prediction)
                
                pos += np.sum(prediction,axis=0).astype(int)
                correct += np.sum(match,axis=0).astype(int)
                total += targets.shape[0]

                # for AUC
                for j in range(10):
                    actual[j] = np.concatenate((actual[j], labels[i][:int(seq_len[i]),j].view(-1).cpu().numpy()))
                    predictions[j] = np.concatenate((predictions[j], outputs[i][:int(seq_len[i]),j].view(-1).cpu().numpy()))


        val_losses[epoch] = running_loss/len(val_loader)
        val_acc[epoch] = correct/total
        val_freq[epoch] = pos / sum(pos)
        val_auc[epoch] = get_aucs(actual, predictions)


if __name__ == '__main__':
    save = not args.nosave
    arch_name = '_'.join(['dim'+str(args.dim), 'heads'+str(args.heads), 'levels'+str(args.levels)])
    opt_name = '_'.join(['lr'+str(args.lr), 'b1_'+str(args.b1), 'b2_'+str(args.b2), 
                        'drop'+str(args.drop), 'l2_'+str(args.l2)])
    if args.nosave: print('WARNING: MODEL AND DATA ARE NOT BEING SAVED')
    elif not args.nosave:
        if not os.path.exists(save_path + arch_name):
            os.mkdir(save_path + arch_name)
        if not os.path.exists(save_path + arch_name + '/' + opt_name):
            os.mkdir(save_path + arch_name + '/' + opt_name)
        else:
            print('WARNING: overwriting existing directory:', arch_name + '/' + opt_name)
    save_path = save_path + arch_name + '/' + opt_name + '/'
    model_name = arch_name + '/' + opt_name

    print('training data setup')
    ''' training data setup '''
    if args.cv == 1 or args.cv == 2 or args.cv == 3:
        valid_indices = list(range(4214))
    elif args.cv == 4:
        valid_indices = list(range(4215))
    else:
        valid_indices = list(range(4213))
    valid_indices = list(range(8283))
    v_weights = get_valid_weights(valid_indices, valid_path, False)
    t_weights = get_train_weights(train_path, False)

    print('init transformer')
    '''Transformer'''
    model = RT(input_size=190, d_model=args.dim, output_size=10, h=args.heads, rnn_type='RNN',
                ksize=args.ksize, n=args.rnn, n_level=args.levels, dropout=args.drop, device=args.device).to(args.device)
    print('criterion')
    criterion = nn.BCELoss(reduction='none')

    print('optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.l2)
    print('scheduler')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=False)


    print('init trackers')
    # loss tracker
    train_losses = np.zeros(args.epochs)
    val_losses = np.zeros(args.epochs)
    # accuracies tracker
    train_acc = np.zeros((args.epochs, 10))
    val_acc = np.zeros((args.epochs, 10))
    # frequency tracker
    train_freq = np.zeros((args.epochs, 10))
    val_freq = np.zeros((args.epochs, 10))
    # AUC traker
    train_auc = np.zeros((args.epochs, 10))
    val_auc = np.zeros((args.epochs, 10))

    # val data same every epoch
    val_data = Dataset(valid_indices, valid_path)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, generator=torch.Generator(device='cuda'))

    start = time.time()
    print('starting...')
    for epoch in range(args.epochs):
        train_loader = make_train_loader(train_path, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, cv=args.cv)
        print('made train loader')
        model.train()
        train(t_weights)
        model.eval()
        valid(v_weights)
        scheduler.step(-np.mean(val_auc[epoch])) # step on avg. AUC plateau

        if args.prog:
            show_prog(epoch, train_losses[epoch], train_acc[epoch], train_auc[epoch],
                        val_losses[epoch], val_acc[epoch], val_auc[epoch], (time.time() - start))

        best_loss = val_losses[epoch] == min(val_losses[:epoch+1])
        best_t_loss = train_losses[epoch] == min(train_losses[:epoch+1])
        best_auc = np.mean(val_auc[epoch]) == max(np.mean(val_auc[:epoch+1],axis=1))

        if save:
            save_prog(model, save_path, train_losses, val_losses, epoch, best_loss, best_t_loss, best_auc)

    # PLOT GRAPHS
    if save:
        auc_plotter(model_name, train_losses,
                val_losses, val_auc, save=save_path, show=False)
    else:
        auc_plotter(model_name, train_losses,
                val_losses, val_auc, save=False, show=True)
    
    if not args.search:
        print('Model:', model_name, 'completed ; ', args.epochs, 'args.epochs', 'in %ds' % (time.time()-start))
        print('min vl_loss: %0.3f at epoch %d' % (min(val_losses), val_losses.argmin()+1))
        print('min tr_loss: %0.3f at epoch %d' % (min(train_losses), train_losses.argmin()+1))
        print('max avg AUC: %0.3f at epoch %d' % (max(np.mean(val_auc[:epoch+1],axis=1)), np.mean(val_auc[:epoch+1],axis=1).argmax()+1))
    else:
        # optim loss, v_loss, t_loss, v_auc, t_auc, runtime, name
        optim_loss = str(-max(np.mean(val_auc[:epoch+1],axis=1)))
        v_loss = str((min(val_losses), val_losses.argmin()+1))
        t_loss = str((min(train_losses), train_losses.argmin()+1))
        v_auc = str((max(np.mean(val_auc[:epoch+1],axis=1)), np.mean(val_auc[:epoch+1],axis=1).argmax()+1))
        t_auc = str((max(np.mean(train_auc[:epoch+1],axis=1)), np.mean(train_auc[:epoch+1],axis=1).argmax()+1))
        runtime = str(time.time()-start)
        name = model_name
        print('&'.join((optim_loss,v_loss,t_loss,v_auc,t_auc, runtime, name)))
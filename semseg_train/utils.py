import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
from torch.autograd import Variable
import tqdm

from callbacks import EarlyStopping

def variable(x, volatile=False):
    #xの型がlistまたはtupleに等しいときTrue
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    #以下一行UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
    #return cuda(Variable(x, volatile=volatile))
    with torch.no_grad():
        return cuda(Variable(x))


def cuda(x):
    #Python3.7以降予約語にasyncが指定されたため以下であると"SyntaxError: invalid syntax"、代わりにnon_blocking
    #return x.cuda(async=True) if torch.cuda.is_available() else x
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x



def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    print(torch.cuda.is_available())
    if model_path.exists():
        #RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        #state = torch.load(str(model_path))
        #state = torch.load(str(model_path), map_location=torch.device("gpu"))
        #RuntimeError: Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu device type at start of device string: 0
        state = torch.load(str(model_path), map_location=torch.device("cuda:0"))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        #プログレスバーの表示
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                #勾配計算をしたいからVariable()で囲む？
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                #パラメータW,Bの勾配値(偏微分)は蓄積してしまうため毎ループで0にする
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                #誤差逆伝搬法
                loss.backward()
                #パラメータ(W,B)更新
                optimizer.step()
                step += 1
                tq.update(batch_size)
                #invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number
                #問題となったコード <class 'torch.Tensor'> は0インデックスが使えないようです
                #losses.append(loss.data[0])
                losses.append(loss.data.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

#EarlyStoppingを追加した学習をしたい場合
def train_callbacks(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None):
    #torch_fix_seed(seed=42)
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)
    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path), map_location=torch.device("cuda:0"))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 100
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    early_stopping = EarlyStopping(patience=15)
    for epoch in range(epoch, n_epochs + 1):
        #torch_fix_seed(seed=42+epoch)
        model.train()
        #random.seed()
        #プログレスバーの表示
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                #勾配計算をしたいからVariable()で囲む？
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                #パラメータW,Bの勾配値(偏微分)は蓄積してしまうため毎ループで0にする
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                #誤差逆伝搬法
                loss.backward()
                #パラメータ(W,B)更新
                optimizer.step()
                step += 1
                tq.update(batch_size)
                #loss.data.item()はfloatでバッチ数分のlossで0.8/1バッチとか
                losses.append(loss.data.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            #save(epoch + 1)
            #if epoch==24:
            #    torch.save({
            #        'model': model.state_dict(),
            #        'epoch': 24,
            #        'step': step,
            #    }, str(root / 'model_{fold}_24epoch.pt'.format(fold=fold)))
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            early_stopping(valid_loss#)
                           , model, epoch+1, step, root / 'model_{fold}_{ep}epoch.pt'.format(fold=fold, ep=epoch))
            if early_stopping.early_stop: 
            #一定epochだけval_lossが最低値を更新しなかった場合、学習終了
                break
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
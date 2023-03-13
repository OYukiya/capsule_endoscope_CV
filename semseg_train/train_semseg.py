import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from validation import validation_binary
from models import UNet, UNet11, UNet16, AlbuNet34#, LinkNet34
from loss import LossBinary, LossWeightBCE, LossBCE, LossJaccard
from dataset import AngyodysplasiaDataset
import utils

from prepare_train_val import get_split
from generate_masks import get_model
from transforms import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        RandomHueSaturationValue,
                        VerticalFlip,
                        NoTransform)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.3, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=1)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=12)
    arg('--limit', type=int, default=None, help='number of images in epoch')
    arg('--n-epochs', type=int, default=500)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--model', type=str, default='UNet16', choices=['UNet', 'UNet11', 'UNet16', 'AlbuNet34'])

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1
    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained=True)
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained=True)
    elif args.model == 'AlbuNet':
        model = AlbuNet34(num_classes=num_classes, pretrained=True)
    elif args.model == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes, pretrained=True)
    else:#ファインチューニング用（モデルの重みの初期値に学習済みモデルを使用）
        model_path = 'angioectasia_paper/UNet16-20220428T113422Z-001/UNet16/model_4.pt'
        model = get_model(model_path, model_type='UNet16')
            
        #print(model)
        # 転移学習用（モデルの重みを固定）
        #model.一覧#pool,encoder,relu,conv1,conv2,conv3,conv4,conv5,center,dec5,dec4,dec3,dec2,dec1,final
        #for param in model.conv1.parameters():
        #    param.requires_grad = False
        #for param in model.conv2.parameters():
        #    param.requires_grad = False
        #for param in model.conv3.parameters():
        #    param.requires_grad = False
        #for param in model.conv4.parameters():
        #    param.requires_grad = False
        #for param in model.conv5.parameters():
        #    param.requires_grad = False

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        
    #損失関数の切替
    loss = LossBinary(jaccard_weight=args.jaccard_weight)
    #loss = LossWeightBCE()
    #loss = LossBCE()
    #loss = LossJaccard()

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, limit=None):
        return DataLoader(
            dataset=AngyodysplasiaDataset(file_names, 
                                            transform=transform, 
                                            limit=limit),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_transform = DualCompose([
        #NoTransform()
        #CenterCrop(256),
        #HorizontalFlip(),
        #VerticalFlip(),
        #Rotate(),
        #ImageOnly(RandomHueSaturationValue()),
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        #NoTransform()
        #CenterCrop(256),
        ImageOnly(Normalize())
    ])
    #データローダの作成
    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, limit=args.limit)
    valid_loader = make_loader(val_file_names, transform=val_transform)
        
    ##transform後の画像表示(batch_size=1にしてから一番最初の画像)
    #import matplotlib.pyplot as plt
    #for i, (inputs, targets) in enumerate(train_loader):#(12,3,256,256)
    #    print(inputs.numpy().shape)
    #    inputs, targets = inputs.numpy()[0].transpose(1, 2, 0), targets[0].numpy().transpose(1, 2, 0)
    #    plt.imshow(inputs.astype('uint8'))#引数がfloatの時範囲は[0..1]でintの時[0..255]
    #    plt.show()
    #    plt.imshow(targets.astype('uint8'))
    #    plt.show()
        
    #'runs/debug/params.json'にコマンド引数の内容を保存する
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    #学習の実行
    utils.train_callbacks(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),#lambda 〇〇:□□の時〇が引数で□が式
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation_binary,
        fold=args.fold
    )


if __name__ == '__main__':
    main()
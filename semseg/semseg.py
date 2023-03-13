import time
import glob
import os
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.autograd import Variable

from models import UNet16

#小腸カプセル内視鏡画像(png)に対して学習済み(重み：ptファイル)セマンティックセグメンテーションモデルへ入力する
#   ※predict_main()を実行するとできる

def cuda(x):
    '''
    GPUが使用可能なら入力画像データをGPUにのせる
    '''
    #Python3.7以降予約語にasyncが指定されたため以下であると"SyntaxError: invalid syntax"、代わりにnon_blocking
    #return x.cuda(async=True) if torch.cuda.is_available() else x
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x

def variable(x, volatile=False):
    '''
    PyTorchで使用可能な型に変換する
    '''
    #xの型がlistまたはtupleに等しいときTrue
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    #以下一行UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
    #return cuda(Variable(x, volatile=volatile))
    with torch.no_grad():
        return cuda(Variable(x))

def get_model(model_path, model_type):
    '''
    学習済みのUNet16の重みを読み込み、UNet16の重みに入れる
    :param model_path:
    :param model_type: 'UNet16'
    :return:
    '''
    model = UNet16(num_classes=1)
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        return model.cuda()
    return model

def mask_overlay(image, mask, color=(0, 255, 0)):
    '''
    出力の二値画像(白が病変、黒が正常)と入力のCE画像を重ね合わせる
    '''
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]    
    return img

def predict_main(in_path='./input_image', out_path='./result', model_path='weight/model_1_20epoch_8.pt', stat_area=300, r_display=0, g_display=255, b_display=0, source_save=True):
    '''
    学習済みモデルによる推論の実行
    
    in_path:入力画像フォルダ(FCNであるため入力サイズは可変、本研究では512×512px)
    out_path:出力画像保存フォルダ
    model_path:学習済みモデルの保存されているフォルダ
    stat_area:出力において、病変と予測された領域の面積がstat_area未満の場合削除される
    r_display, g_display, b_display:セグメンテーション領域を何色にするか
    source_save:True  ⇒  元画像を保存
    '''
    file_names_ = [
                    in_path
                    ]
    save_file_names = [
                    out_path
                    ]
    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = get_model(f'{model_path}', model_type='UNet16')#モデルの読み込み
    model_name_only = os.path.splitext(os.path.basename(model_path))[0]
    for j, file_names in enumerate(file_names_):#フォルダ数(1回)ループ
        read_file_name = glob.glob(f'{file_names}/*.png')
        read_file_name_only = [os.path.splitext(os.path.basename(p))[0] for p in read_file_name]
        os.makedirs(f'{save_file_names[j]}/{model_name_only}', exist_ok=True)
        if source_save:
            os.makedirs(f'{save_file_names[j]}/{model_name_only}/source', exist_ok=True)
        print(file_names)
        start_ = time.time()
        for i in range(len(read_file_name)):#画像枚数分ループ
            img = cv2.imread(str(read_file_name[i]))#入力画像の読み込み
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_img = torch.unsqueeze(variable(img_transform(img), volatile=True), dim=0)
            mask = model(input_img)#推論の実行(maskは二値画像で白⇒病変、黒⇒それ以外)
            mask_array = mask.data[0].cpu().numpy()[0]
            mask_array = np.where(mask_array > 0, np.uint8(255), np.uint8(0))
            retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array)#推論後の領域毎の情報を取得
            if mask_array.sum() >= stat_area:
                count_small_blob_num = 0 #stat_area[px]以下の領域の数
                for i2, row in enumerate(stats):
                    #print(f"label {i}")
                    #print(f"* topleft: ({row[cv2.CC_STAT_LEFT]}, {row[cv2.CC_STAT_TOP]})")
                    #print(f"* size: ({row[cv2.CC_STAT_WIDTH]}, {row[cv2.CC_STAT_HEIGHT]})")
                    #print(f"* area: {row[cv2.CC_STAT_AREA]}")
                    if (row[cv2.CC_STAT_AREA]<stat_area) & (i2!=0):#領域ごとで面積がstat_area[px]未満の場合に黒くする
                        count_small_blob_num += 1
                        mask_array = np.where(labels[:] == i2, 0, mask_array)
                count_blob_num = (retval - 1) - count_small_blob_num #領域の個数
                if count_blob_num>0:#領域の個数が1個以上で画像を保存する
                    if source_save:
                        cv2.imwrite(f'{save_file_names[j]}/{model_name_only}/source/{read_file_name_only[i]}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    g_display = 1 if g_display==0 else g_display #g_displayの値が0だとなぜか色が表示されないので0なら1にする
                    saveimg = mask_overlay(img, (mask_array > 0).astype(np.uint8), color=(r_display, g_display, b_display))
                    cv2.imwrite(f'{save_file_names[j]}/{model_name_only}/{read_file_name_only[i]}.png', cv2.cvtColor(saveimg, cv2.COLOR_BGR2RGB))
        print(time.time() - start_)

if __name__=="__main__":
    predict_main(in_path='./input_image', out_path='./result', model_path='weight/model_1_20epoch_8.pt', stat_area=300, r_display=0, g_display=255, b_display=0, source_save=True)
    '''
    引数
    in_path:入力画像フォルダ(FCNであるため入力サイズは可変、本研究では512×512px)
    out_path:出力画像保存フォルダ
    model_path:学習済みモデルの保存されているフォルダ
    stat_area:出力において、病変と予測された領域の面積がstat_area未満の場合削除される
    r_display, g_display, b_display:セグメンテーション領域を何色にするか
    source_save:True  ⇒  元画像を保存
    '''
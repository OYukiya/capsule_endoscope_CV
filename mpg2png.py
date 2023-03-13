import os, datetime, glob, re
import cv2
import numpy as np
import pandas as pd
import csv
import sys

#RAPIDから保存した小腸カプセル内視鏡動画(mpg)を画像(png)にして保存
#   ※simplemain()を実行するとできる

def matching(x, y, image, num):
    '''
    パターンマッチングの実行
    '''
    w = 17
    h = 24
    _time = 0
    try:
        for i in range(10, -1, -9):
            img = image[y:y+h, x:x+w]
            mat = []
            for n in num:
                match_result = cv2.matchTemplate(img, n, cv2.TM_CCOEFF_NORMED)
                mat.append(match_result.max())
            t = np.argmax(mat)
            _time += t*i
            x += 17
    except Exception as e:
        print(f'numbers/0～9.pngが見つかりません。CE画像の時間読み取り用の0～9.pngをnumbersフォルダに入れて適切に配置してください。\nエラー内容：{e}')
        sys.exit()
    return _time

def readtime(img):
    '''
    パターンマッチングによりCE画像の時刻を読み取る
    '''
    _time = []
    x = 5
    y = 7
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numpath = glob.glob('./numbers/*.png')
    num = []
    for path in numpath:
        n = cv2.imread(path)
        n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        num.append(n)

    for i in range(3):
        t = matching(x, y, img, num)
        _time.append(t)
        x += 44

    return _time[0], _time[1], _time[2]

def fiximg(img, mask):
    '''
    画像を512×512にして淵の白文字を黒くする
    '''
    return img[32:-32, 32:-32, :]*mask

def mpg2png(movpath, folder, img_save, inp_save, imgpath=None, inppath=None):
    '''
    動画を画像にして保存
    (画像名：CE画像の時間で保存、時間が重複している場合は、名前の最後が"_00""_01"などになる)
    '''
    if inp_save:
        try:
            mask = cv2.imread('./mask.png')
            mask = np.where(mask < 100, 0, 1)
        except Exception as e:
            print(f'mask.pngが見つかりません。CE画像の名前や時間を黒塗りするためのmask.pngが読み込めませんでした。パスを確認してください。\nエラー内容：{e}')
            sys.exit()
    hours = []
    minute = []
    second = []
    total_second = []
    i = 0
    tmp = None
    tmps = 0
    cap = cv2.VideoCapture(movpath)
    if(cap.isOpened()):
        ret, frame = cap.read()
        while(ret):#画像枚数分ループ
            if(i%5 == 0):#CE動画は5枚連続で同じ画像のため5回に一回保存
                h, m, s = readtime(frame)
                hours.append(h)
                minute.append(m)
                second.append(s)
                ts = h*3600 + m*60 + s
                total_second.append(ts)

                if (ts < tmps):
                    print(f'failed pattern matching.CE画像から時刻の読み取りに失敗しました。')
                    sys.exit()

                ce_time = '{0:02}{1:02}{2:02}'.format(h, m, s)
                if(ce_time == tmp):
                    n += 1
                else:
                    n = 0
                if img_save:
                    cv2.imwrite(os.path.join(imgpath, ce_time + '_{:02}.png'.format(n)), frame)
                if inp_save:
                    frame = fiximg(frame, mask)
                    cv2.imwrite(os.path.join(inppath, ce_time + '_{:02}.png'.format(n)), frame)
                tmp = ce_time
                tmps = ts
            i += 1
            ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if img_save or inp_save:
        #各画像に書かれている内視鏡の時間をcsvに保存
        df = pd.DataFrame({'Hours': hours, 'Minute': minute, 'Second': second, 'Total_Second': total_second})
        df.to_csv(os.path.join(folder, 'time.csv'))

def simplemain(mpg_paths='./mpg', out_path='./output', img_save=True, inp_save=True):
    '''
    PillCamSB3動画(COLON2非対応)を画像に変換(mpg2pngの実行)と保存するフォルダを生成
    (元画像保存先：out_path/現在の時刻/動画の名前/image/*.png)
    (トリミング後画像保存先：out_path/現在の時刻/動画の名前/input_image/*.png)
    ※time.csvに各画像の時間を書き込む
    args
    mpg_paths:動画を含むフォルダ
    out_path:動画を画像変換後の保存先
    img_save:動画を画像変換後に保存するか
    inp_save:動画を画像変換後にトリミングと白塗りして保存するか
    '''
    mpgpaths = glob.glob(f'{mpg_paths}/*')
    mpgpaths = sorted([p for p in mpgpaths if re.search('/*\.(avi|mp4|mov|mpg|wmv|flv)', str(p))])
    if len(mpgpaths)==0:
        print(f'読み込んだ動画{len(mpgpaths)}本。動画が読み込まれていません。パスの確認やフォルダ名とファイル名に日本語を含まないようにしてください。')
        sys.exit()
    print(f'確認（動画数{len(mpgpaths)}。{mpgpaths}\n以上の動画を画像にして保存します。')
    outpath = os.path.join(out_path, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    imgpath, inppath = None, None
    for mpgpath in mpgpaths:
        mpgname = os.path.splitext(os.path.basename(mpgpath))[0]
        path = os.path.join(outpath, mpgname)
        if img_save:
            imgpath = os.path.join(outpath, mpgname, 'image')
            os.makedirs(imgpath, exist_ok=True)
        if inp_save:
            inppath = os.path.join(outpath, mpgname, 'input_image')
            os.makedirs(inppath, exist_ok=True)

        mpg2png(mpgpath, path, img_save, inp_save, imgpath=imgpath, inppath=inppath)

if __name__ == '__main__':
    simplemain(mpg_paths='./mpg', out_path='./output', img_save=True, inp_save=True)
    #※mask/pngとnumbers/0から9まで.pngを適切に配置する
    #引数
    #mpg_paths:動画を含むフォルダ
    #out_path:動画を画像変換後の保存先
    #img_save:動画を画像変換後に保存するか
    #inp_save:動画を画像変換後にトリミングと白塗りして保存するか
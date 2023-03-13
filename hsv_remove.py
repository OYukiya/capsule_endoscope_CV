import time
import copy
import os
import glob
import numpy as np
import pandas as pd
import cv2

#複数の画像にHSV条件による画像の削減を行い削減後の画像を保存
#    ※simplemain()の引数を設定して使用するとできる

def jugde_bleeding_hsv(img, h_low=20, h_high=200, s_low=120, v_low=5, opening_noise_removing=True, opening_kernel=5, height=512, width=512):
    '''
    height×widthのカラー内視鏡画像(numpy.ndarray)一枚に対しHSV条件に当てはまる画素数を返す
    流れ：HSV条件に当てはまる座標の取得⇒(オープニング処理でノイズ除去)⇒条件に当てはまる座標の個数(red_area_px)と条件に当てはまる画素を白にしたサイズがheight×widthの画像(red_area)を返す
    '''
    img_hsv = copy.deepcopy(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    red_area = np.zeros((height, width, 1), dtype=np.uint8)
    # 条件に当てはまるとき、red_area(真っ黒な画像)のその領域を255にする
    red_area[:,:,0] = np.where((img_hsv[:,:,0]>=h_low) & (img_hsv[:,:,0]<=h_high) & (img_hsv[:,:,1]>=s_low) & (img_hsv[:,:,2]>=v_low), 255, 0)
    if opening_noise_removing:
        # 出血判定した領域にオープニング処理することで小さい領域を消去
        kernel = np.ones((opening_kernel, opening_kernel),np.uint8)
        red_area = cv2.morphologyEx(red_area, cv2.MORPH_OPEN, kernel)
    else:
        red_area = np.reshape(red_area, (height, width))
    # 全出血領域の画素数の合計
    red_area_px = cv2.countNonZero(red_area)#ほぼ0s
    return red_area_px, red_area

def simplemain(png_path='data_all_red/angio_bleeding', out_path='', h_low=0, h_high=15, s_low=205, v_low=156, range360degree_100percent=True, opening_noise_removing=True, opening_kernel=5, height=512, width=512, stat_area=300, source_save=False, r_display=0, g_display=255, b_display=0):
    '''
    height×widthのカラー内視鏡画像に対しHSV条件に当てはまる画像を保存
    (保存先：out_path/png_path/-h-_-s_-v(_360degree100percent_opening_kernel-)/.png)
    args
    png_path:HSV条件を適用させる画像が入ったフォルダ
    out_path:HSV適用後の画像を保存するフォルダ
    h_low, h_high: "□<=H<=□" の□の部分の条件
    s_low: "□<=S" の□の部分の条件
    v_low: "□<=V" の□の部分の条件
    range360degree_100percent:True  ⇒  h_low, h_high, s_low, v_lowの条件で範囲をhが0-360°,sとvが0-100%とするとき
                                False  ⇒  h_low, h_high, s_low, v_lowの条件で範囲を0-255とするとき(cv2.cvtColorで引数がcv2.COLOR_BGR2HSV_FULL)
    opening_noise_removing:True  ⇒  HSV条件適用後にオープニング処理によるノイズ除去をする
    opening_kernel:opening_noise_removingがTrueのときのみ使用。オープニング処理のカーネルサイズ
    height, width:入力画像サイズ
    stat_area:これ未満の領域の面積は削除される
    source_save:True  ⇒  HSV条件に当てはまるときその元画像を保存
    r_display, g_display, b_display:HSV条件に当てはまる部分を何色にするか
    '''
    pngpath_name = glob.glob(f'{png_path}/*.png')
    pngpath_name_only = [os.path.splitext(os.path.basename(p))[0] for p in pngpath_name]
    outpath_only = os.path.splitext(os.path.basename(png_path))[0]
    if opening_noise_removing:
        if range360degree_100percent:
            outpath_save = os.path.join(out_path, png_path, f'{h_low}h{h_high}_{s_low}s_{v_low}v_360degree100percent_opening_kernel{opening_kernel}')
        else:
            outpath_save = os.path.join(out_path, png_path, f'{h_low}h{h_high}_{s_low}s_{v_low}v_opening_kernel{opening_kernel}')
    else:
        if range360degree_100percent:
            outpath_save = os.path.join(out_path, png_path, f'{h_low}h{h_high}_{s_low}s_{v_low}v_360degree100percent')
        else:
            outpath_save = os.path.join(out_path, png_path, f'{h_low}h{h_high}_{s_low}s_{v_low}v')
    os.makedirs(outpath_save, exist_ok=True)
    if source_save:
        os.makedirs(os.path.join(outpath_save, f'source'), exist_ok=True)
    if range360degree_100percent:#Hが0-360°でSとVが0-100%の範囲を0-255にする
        print(f'({h_low}≦H≦{h_high}[°], {s_low}≦S[%], {v_low}≦V[%])pixels, each areas＜{stat_area}')
        h_low = 255*h_low/360
        h_high = 255*h_high/360
        s_low = 255*s_low/100
        v_low = 255*v_low/100
    else:
        print(f'({360*h_low/255}≦H≦{360*h_high/255}[°], {100*s_low/255}≦S[%], {100*v_low/255}≦V[%])pixels, each areas＜{stat_area}')
    start_ = time.time()
    for i in range(0,len(pngpath_name)):
        # 出血判定したい画像の読み込み
        img = cv2.imread(pngpath_name[i])
        for_red_area = copy.deepcopy(img)
        red_area_px, red_area = jugde_bleeding_hsv(for_red_area, h_low=h_low, h_high=h_high, s_low=s_low, v_low=v_low, opening_noise_removing=opening_noise_removing, opening_kernel=opening_kernel, height=height, width=width)
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(red_area)#ラベル数、ラベリング結果、各ラベルの構造情報(左上の x 座標, 左上の y 座標, 幅, 高さ, 面積)、重心
                    
        if red_area_px>=stat_area:#なくてもいいが処理の回数を減らし実行時間が少し早くなる（特にopeningなしだと）
            count_small_area_num = 0 #閾値未満の領域の個数
            for k, row in enumerate(stats):
                ##一画像内の領域の個数とその情報
                #print(f"label {i}")#i番目の領域(0番目は画像全体)
                #print(f"* topleft: ({row[cv2.CC_STAT_LEFT]}, {row[cv2.CC_STAT_TOP]})")#i番目の領域の座標
                #print(f"* size: ({row[cv2.CC_STAT_WIDTH]}, {row[cv2.CC_STAT_HEIGHT]})")#i番目の領域の幅と高さ
                #print(f"* area: {row[cv2.CC_STAT_AREA]}")#i番目の領域の面積
                if (row[cv2.CC_STAT_AREA]<stat_area) & (k!=0):#領域ごと閾値未満で黒塗
                    count_small_area_num += 1
                    red_area = np.where(labels[:] == k, 0, red_area)
            count_area_num = (retval - 1) - count_small_area_num #領域数/1枚
            if count_area_num>0:#領域の個数が1個以上の場合に画像を保存
                img_res = copy.deepcopy(img)
                img_res[:,:,1] = np.where(red_area!=0, g_display, img_res[:,:,1])
                img_res[:,:,0] = np.where(red_area!=0, b_display, img_res[:,:,0])
                img_res[:,:,2] = np.where(red_area!=0, r_display, img_res[:,:,2])
                if source_save:
                    cv2.imwrite(f'{outpath_save}/source/{pngpath_name_only[i]}.png', img)#元画像
                cv2.imwrite(f'{outpath_save}/{pngpath_name_only[i]}.png', img_res)##色条件⇒(オープニング)⇒領域ごとに閾値未満を消した画像
    print(f'実行終了、色の条件による削減画像保存先:{outpath_save}\n\n所要時間:\n{time.time() - start_}s')
if __name__ == '__main__':
    simplemain(png_path='data_all_red/angio_bleeding', out_path='', h_low=0, h_high=21, s_low=80, v_low=22, range360degree_100percent=True, opening_noise_removing=True, opening_kernel=5, height=512, width=512, stat_area=200, source_save=True, r_display=0, g_display=255, b_display=255)
    #引数
    #png_path:HSV条件を適用させる画像が入ったフォルダ
    #out_path:HSV適用後の画像を保存するフォルダ
    #h_low, h_high: "□<=H<=□" の□の部分の条件
    #s_low: "□<=S" の□の部分の条件
    #v_low: "□<=V" の□の部分の条件
    #range360degree_100percent:True  ⇒  h_low, h_high, s_low, v_lowの条件で範囲をhが0-360°,sとvが0-100%とするとき
    #                          False  ⇒  h_low, h_high, s_low, v_lowの条件で範囲を0-255とするとき(cv2.cvtColorで引数がcv2.COLOR_BGR2HSV_FULL)
    #opening_noise_removing:True  ⇒  HSV条件適用後にオープニング処理によるノイズ除去をする
    #opening_kernel:opening_noise_removingがTrueのときのみ使用。オープニング処理のカーネルサイズ
    #height, width:入力画像サイズ
    #stat_area:これ未満の領域の面積は削除される
    #source_save:True  ⇒  HSV条件に当てはまるときその元画像を保存
    #r_display, g_display, b_display:HSV条件に当てはまる部分を何色にするか
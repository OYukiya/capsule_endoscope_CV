import copy
import os
import glob
import time
import optuna
import numpy as np
import pandas as pd
import cv2

#Optunaを使用してHSV条件の組合せを試し各組合せに対してF値を算出する
#※optuna_hsv()を実行するとできる

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

def count_red(abnormal_path='data_all_red/angio_bleeding', normal_path='data_all_red/normal', h_low=20, h_high=200, s_low=120, v_low=5, opening_noise_removing=True, opening_kernel=5, height=512, width=512, save_result=True, stat_area=300):
    '''
    abnormal_path内の病変画像とnormal_path内の正常画像に対しHSV条件に当てはまる画像数を算出してF値を算出する。1-(F値)(optim_func)を返す。
    流れ：HSV条件に当てはまる座標の取得⇒(オープニング処理でノイズ除去)⇒条件に当てはまる座標の個数(red_area)と条件に当てはまる画素を白にしたサイズがheight×widthの画像(b)を返す
    '''
    count_num = [0, 0]#HSVに当てはまる[病変, 正常]画像の枚数をカウントしていく
    read_abnormal_file_name = glob.glob(f'{abnormal_path}/*.png')
    read_normal_file_name = glob.glob(f'{normal_path}/*.png')
    file_names_ = [
                    read_abnormal_file_name,#本研究では784枚
                    read_normal_file_name#本研究では3,200枚
                    ]
    for j in range(2):#j=0は病変画像、j=1は正常画像
        for i in range(0,len(file_names_[j])):#画像の枚数ループ
            # 出血判定したい画像の一枚読み込み
            img = cv2.imread(file_names_[j][i])
            #HSV条件に当てはまる全画素数(red_area_px)とその画素を白にした画像(red_area)
            red_area_px, red_area = jugde_bleeding_hsv(img, h_low=h_low, h_high=h_high, s_low=s_low, v_low=v_low, opening_noise_removing=opening_noise_removing, opening_kernel=opening_kernel, height=height, width=width)
            retval, labels, stats, _ = cv2.connectedComponentsWithStats(red_area)#ラベル数、ラベリング結果、各ラベルの構造情報(左上の x 座標, 左上の y 座標, 幅, 高さ, 面積)、重心
                    
            if red_area_px>=stat_area:#なくてもいいが処理数を減らし実行時間が少し短縮（特にopenningなしだと）
                count_small_area_num = 0 #閾値未満の領域の個数
                for k, row in enumerate(stats):#画像一枚の中にある領域の個数分ループ
                    ##一画像内の領域の個数とその情報
                    #print(f"label {i}")#i番目の領域(0番目は画像全体)
                    #print(f"* topleft: ({row[cv2.CC_STAT_LEFT]}, {row[cv2.CC_STAT_TOP]})")#i番目の領域の座標
                    #print(f"* size: ({row[cv2.CC_STAT_WIDTH]}, {row[cv2.CC_STAT_HEIGHT]})")#i番目の領域の幅と高さ
                    #print(f"* area: {row[cv2.CC_STAT_AREA]}")#i番目の領域の面積
                    if (row[cv2.CC_STAT_AREA]<stat_area) & (k!=0):#領域ごと閾値未満で黒塗
                        count_small_area_num += 1
                        red_area = np.where(labels[:] == k, 0, red_area)
                count_area_num = (retval - 1) - count_small_area_num #領域数
                if count_area_num>0:#領域の個数が1個以上の場合に病変画像としてカウント+1
                    count_num[j] += 1
    precision = count_num[0]/(count_num[0]+count_num[1]+0.0000000000001)#適合率
    recall = count_num[0]/len(read_abnormal_file_name)#再現率（感度）
    optim_func = 1 - (2 * precision * recall / (precision + recall + 0.0000000000001)) #1 - F値
    print(f'count:[{count_num[0]}/{len(read_abnormal_file_name)} {count_num[1]}/{len(read_normal_file_name)}]')
    #if save_result:
    #    #各変数をcsvに保存
    #    df = pd.DataFrame({'h_low': [h_low], 'h_high': [h_high], 's_low': [s_low], 'v_low': [v_low], 'abnormal': [count_num[0]], 'normal': [count_num[1]], '1-F': [optim_func]})
    #    df.to_csv(os.path.join('', 'result.csv'))
    return optim_func

def objective_args(abnormal_path='data_all_red/angio_bleeding', normal_path='data_all_red/normal', h_low_min=12, h_low_max=20, h_high_min=245, h_high_max=255, s_low_min=200, s_low_max=230, v_low_min=10, v_low_max=80, opening_noise_removing=True, opening_kernel=5, height=512, width=512, save_result=True, stat_area=300):
    '''
    Optunaを使用して探索するため、各変数の範囲の設定
    '''
    def objective(trial):
        x1 = trial.suggest_int('h_low', h_low_min, h_low_max)
        x2 = trial.suggest_int('h_high', h_high_min, h_high_max)
        x3 = trial.suggest_int('s_low', s_low_min, s_low_max)
        x4 = trial.suggest_int('v_low', v_low_min, v_low_max)
        start_ = time.time()
        optim_func = count_red(abnormal_path=abnormal_path, normal_path=normal_path, h_low=x1, h_high=x2, s_low=x3, v_low=x4, opening_noise_removing=opening_noise_removing, opening_kernel=opening_kernel, height=height, width=width, save_result=save_result, stat_area=stat_area)
        print(f'time:{time.time() - start_}s')
        return optim_func
    return objective

def optuna_hsv(abnormal_path='data_all_red/angio_bleeding', normal_path='data_all_red/normal', h_low_min=12, h_low_max=20, h_high_min=245, h_high_max=255, s_low_min=200, s_low_max=230, v_low_min=10, v_low_max=80, range360degree_100percent=True, n_trials=500, opening_noise_removing=False, opening_kernel=5, height=512, width=512, save_result=True, save_result_path='', stat_area=300):
    '''
    Optunaを使用して、TPESamplerで探索
    '''
    if range360degree_100percent:#入力したh_low, h_high, s_low, v_lowの条件で範囲がhが0-360°,sとvが0-100%のとき0-255に変換する
        print(f'h_low:{h_low_min}～{h_low_max}[°], h_high:{h_high_min}～{h_high_max}[°], s_low:{s_low_min}～{s_low_max}[%], v_low:{v_low_min}～{v_low_max}[%]')
        h_low_min = (255*h_low_min)//360
        h_low_max = (255*h_low_max)//360
        h_high_min = (255*h_high_min)//360
        h_high_max = (255*h_high_max)//360
        s_low_min = (255*s_low_min)//100
        s_low_max = (255*s_low_max)//100
        v_low_min = (255*v_low_min)//100
        v_low_max = (255*v_low_max)//100
        print(f'h_low:{h_low_min}～{h_low_max}, h_high:{h_high_min}～{h_high_max}, s_low:{s_low_min}～{s_low_max}, v_low:{v_low_min}～{v_low_max}[0-255]')
    else:
        print(f'h_low:{h_low_min}～{h_low_max}, h_high:{h_high_min}～{h_high_max}, s_low:{s_low_min}～{s_low_max}, v_low:{v_low_min}～{v_low_max}[0-255]')
        print(f'h_low:{(360*h_low_min)/255}～{(360*h_low_max)/255}[°], h_high:{(360*h_high_min)/255}～{(360*h_high_max)/255}[°], s_low:{(100*s_low_min)/255}～{(100*s_low_max)/255}[%], v_low:{(100*v_low_min)/255}～{(100*v_low_max)/255}[%]')
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective_args(abnormal_path=abnormal_path, normal_path=normal_path, h_low_min=h_low_min, h_low_max=h_low_max, h_high_min=h_high_min, h_high_max=h_high_max, s_low_min=s_low_min, s_low_max=s_low_max, v_low_min=v_low_min, v_low_max=v_low_max, opening_noise_removing=opening_noise_removing, opening_kernel=opening_kernel, height=height, width=width, save_result=save_result, stat_area=stat_area), n_trials=n_trials)
    if save_result:
        study.trials_dataframe().to_csv(f'{save_result_path}/study_history.csv')
    print(f'params:{study.best_params}')

if __name__ == '__main__':
    optuna_hsv(abnormal_path='data_all_red/angio_bleeding', normal_path='data_all_red/normal', h_low_min=12, h_low_max=20, h_high_min=245, h_high_max=255, s_low_min=200, s_low_max=230, v_low_min=10, v_low_max=80, range360degree_100percent=False, opening_noise_removing=True, opening_kernel=5, height=512, width=512, save_result=True, save_result_path='', stat_area=300, n_trials=2)
    #引数
    #abnormal_path:HSV条件を適用させる病変(血管拡張または出血)画像が入ったフォルダ
    #normal_path:HSV条件を適用させる正常画像が入ったフォルダ
    #out_path:HSV適用後の画像を保存するフォルダ
    #h_low_min, h_low_max, h_high_min, h_high_max: "□<=H<=〇" の□と〇の部分の条件の範囲
    #s_low_min, s_low_max: "□<=S" の□の部分の条件の範囲
    #v_low_min, v_low_max: "□<=V" の□の部分の条件の範囲
    #range360degree_100percent:True  ⇒  h_low, h_high, s_low, v_lowの条件で範囲をhが0-360°,sとvが0-100%とするとき
    #                          False  ⇒  h_low, h_high, s_low, v_lowの条件で範囲を0-255とするとき(cv2.cvtColorで引数がcv2.COLOR_BGR2HSV_FULL)
    #opening_noise_removing:True  ⇒  HSV条件適用後にオープニング処理によるノイズ除去をする
    #opening_kernel:opening_noise_removingがTrueのときのみ使用。オープニング処理のカーネルサイズ
    #height, width:入力画像サイズ
    #save_result:Optunaの最適化結果を保存するかどうか
    #save_result_path:結果保存先
    #stat_area:これ未満の領域の面積は削除される
    #n_trials:hsvの組合せを使用した最適化を何回行うか
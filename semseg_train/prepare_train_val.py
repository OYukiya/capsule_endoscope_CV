from dataset import data_path
from sklearn.model_selection import KFold
import numpy as np


def get_split(fold, num_splits=5):
    #Pathオブジェクトに対して"/"を使用するとパスが連結される（.joinpath()メソッドでも同様）
    #train_path = data_path / 'train' / 'angyodysplasia' / 'images'
    #train_path = data_path / 'train' / 'KvasirSEG' / 'images'
    #train_path = data_path / 'train' / 'bleeding' / 'images'
    train_path = data_path / 'train' / 'angio_bleeding_No128hsv_aug810_256' / 'images'

    train_file_names = np.array(sorted(list(train_path.glob('*'))))

    #データをK分割しfold番目のtrainとtestの組み合わせを使用する：n_splits分割、random_stateは乱数のシードを指定できる(各クラスの分割のランダム性を制御)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    ids = list(kf.split(train_file_names))

    train_ids, val_ids = ids[fold]

    if fold == -1:
        return train_file_names, train_file_names
    else:
        return train_file_names[train_ids], train_file_names[val_ids]


if __name__ == '__main__':
    ids = get_split(0)

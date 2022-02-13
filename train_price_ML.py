# x - szereg czasowy oceny
# y - binarny wynik zmiany ceny

import os
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Activation, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight  # potrzebne do ustalenia wag celem zbalansowaniu zbioru danych
# from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
np.random.seed(4)
from collections import deque
import random

from rich import print as rprint
import diagnostics_utils as du

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


#PONIŻSZE ZMIENNE NALEŻY GDZIEŚ UKRYĆ, BO SĄ WYWOŁYWANE ZAWSZE W MAINIE!
MODEL_NAME = 'Price_ML'
BATCH_SIZE = 16                         #int podzielny przez 8 
EPOCH = 7     

HISTORY_POINTS = 15                     # liczba punktów na krzywej nastroju brana pod uwagę, służąca za wejście
UP_PRICE_CHANGE = 1.05                  # np 1.01
DOWN_PRICE_CHANGE = 0.95
FUTURE_PERIOD_PREDICT = 2               # [2h]


def scores_series_model(train_ds=None):
    rprint('[italic red] Defining the model... [/italic red]')
    if train_ds:
        X = train_ds[0]
        X_shape = X.shape[1:]
    else:
        X_shape = (HISTORY_POINTS, 1)    #ręcznie ustawiany rozmiar

    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=X_shape))
    model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=1, activation='softmax'))#"sigmoid"))
    return model


def scores_series_model2(train_ds=None):
    rprint('[italic red] Defining the model... [/italic red]')
    if train_ds:
        X = train_ds[0]
        X_shape = X.shape[1:]
    else:
        X_shape = (HISTORY_POINTS, 1)    #ręcznie ustawiany rozmiar

    model = Sequential()
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.35, return_sequences=True, input_shape=X_shape))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=128,  dropout=0.1, recurrent_dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())

    model.add(LSTM(units=128,  dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=3, activation='softmax'))
    return model


def get_scores(dirs, filepath_scores, price_file_name):
    scores, _ = du.open_scores(filepath=filepath_scores)
    crypto_prices = du.get_crypto_prices(os.path.join(dirs['prices'], price_file_name))
    df = pd.merge(scores, crypto_prices, how='inner', left_index=True, right_index=True)
    crypto_scores_weight = df['count'].mean()
    df = df[['mean','close']]   # pozbywamy się 'count' oraz 'info' 
    return df, crypto_scores_weight
    # Getting the files with scores and prices


def get_timeseries_batches(df, columns_to_use = ['mean'], target=None, seq_len=HISTORY_POINTS):
    # columns_to_use - ['mean'] / ['close'] / ['mean','close']
    if target is None:
        target = [0]*len(df)
    sequential_data = []
    prev_days = deque(maxlen=seq_len)  # Deque wyrzuca wartość z końca po przekroczeniu limitu
    
    df = df.sample(frac=1).reset_index(drop=True)   #bo index musi stanowić liczby 0,1,2,...
    for index, row in df.iterrows():
        prev_days.append(list(row[columns_to_use])) # Dodaje jako element listę, stanowiącą cały rząd DFa.
        if len(prev_days) == seq_len:    # Odczekuje seq_len iteracji, a następnie już w każdej iteracji wchodzi do ifa
            sequential_data.append([np.array(prev_days), target[index]])

    random.shuffle(sequential_data)  # (zmniejsza acc o 6pkt %-owych)
    return sequential_data


def preprocess_scores(df):
    # Wyjście:
    df['future_close'] = df['close'].shift(-FUTURE_PERIOD_PREDICT)
    target = []
    for index, row in df.iterrows():
        if row['future_close']/row['close'] > UP_PRICE_CHANGE:
            target.append(1)
        elif row['future_close']/row['close'] < DOWN_PRICE_CHANGE:
            target.append(2)
        else:
            target.append(0)
    target = target_to_onehot(np.array(target))

    # Skalowanie: 
    for col in df.columns:      # ['mean','close']:
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
        df[col] = preprocessing.scale(df[col].values) #przeskalowanie wartości (rozkład gaussa, nie do <0,1>)
    df.dropna(inplace=True)

    # Szeregi czasowe:
    sequential_data = get_timeseries_batches(df, target=target) 
    return sequential_data
    # Utworzenie zbioru danych. Wartosci wyjsciowe: df['series'] tabela wejsc, db[buy_now']


def target_to_onehot(target_npa):
    print('Wyjścia: ', np.unique(target_npa, return_counts=True)[1])
    target_npa = target_npa.reshape(len(target_npa), 1)
    target_oh = preprocessing.OneHotEncoder(sparse=False).fit_transform(target_npa) #[1 0 0] / [0 1 0] / [0 0 1]
    return target_oh
    # Zmienienie kodowania wyjścia na OneHot


def to_numpy_tuple(ds):
    X = [x[0] for x in ds]
    Y = [y[1] for y in ds]
    X = [x.astype('float32') for x in X]    #np.asarray(x).
    Y = [y.astype('float32') for y in Y]
    X = np.array(X)     #ponieważ powyżej utworzona jest zwykła lista
    Y = np.array(Y)
    return (X, Y) 
    #Wpisanie danych z dataframe do tabel numpy 


def standarize_scores_values(df):
    mean_max = df.nlargest(1, 'mean')['mean'].mean()    #największa wartość w kolumnie 'mean'
    mean_min = df.nsmallest(1, 'mean')['mean'].mean()   #najmniejsza wartość w kolumnie 'mean'
    df['mean'] = (df['mean'] - mean_min )/(mean_max - mean_min) #wartości teraz są w przedziale <0, 1>

    price_max = df.nlargest(1, 'close')['close'].mean()
    price_min = df.nsmallest(1, 'close')['close'].mean()
    df['close'] = (df['close'] - price_min) /(price_max - price_min)   #jak wyżej. przedział <0, 1>
    return df
    #Zmiana cen i scorów tak aby przyjmowały wartości od 0 do 1


def setting_optimizer():
    # 'value clipping' zaradza problemowi 'explosing gradient'

    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    # optimizer = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-6)  
    # optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
    # gvs = optimizer.compute_gradients('loss')
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # train_op = optimizer.apply_gradients(capped_gvs)
    # return train_op
    return tf.keras.optimizers.Adam(clipvalue=0.5, learning_rate=1e-3,decay=1e-6)


def train(dirs, model, train_ds, val_ds):
    rprint(f'[italic red] Training the model, on DS of length {len(train_ds[0])}  [/italic red]')
    # '({int(len(train_ds[0])/3)} per target value)...'
    X = train_ds[0]
    Y = train_ds[1]

    log_name = f"{HISTORY_POINTS}seq-{UP_PRICE_CHANGE:.3f}prc_change-{int(FUTURE_PERIOD_PREDICT*2)}H_pred-{int(time.time())}"
    tensorboard = TensorBoard(log_dir = os.path.join(dirs['dir0'], 'machine_learning', MODEL_NAME, 'logs', log_name))

    checkpoint_dir = os.path.join(dirs['dir0'], 'machine_learning', MODEL_NAME, 'models')
    checkpoint_name = "PriceLSTM-{epoch:02d}ep-{val_acc:.3f}accu"
    checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, checkpoint_name),
                                monitor='val_acc',
                                save_weights_only=True,
                                save_best_only=True, 
                                mode='max',
                                verbose=1)

    model.fit(X,Y, BATCH_SIZE, EPOCH, 
            validation_data=val_ds,
            callbacks=[tensorboard, checkpoint])


#==main
def perform_machine_learning(dirs, selected_files_tuples):
    rprint(f'[italic red] Scores per timeseries: {HISTORY_POINTS} [/italic red]')
    # df_train = pd.DataFrame()
    # df_val = pd.DataFrame()
    train_set = []
    val_set = []
    for ft in selected_files_tuples:
        scores_fpath = os.path.join(dirs['selected_scores'], ft[0])
        new_df, crypto_scores_weight = get_scores(dirs, scores_fpath, ft[1])

        train_new, val_new = train_test_split(new_df, test_size=0.2)
        train_new = preprocess_scores(train_new)
        val_new = preprocess_scores(val_new)
        train_set = [*train_set, *train_new]
        val_set = [*val_set, *val_new]
        # df_train = pd.concat([df_train, new_df_train])
        # df_train = pd.concat([df_train, new_df_val])

        # crypto_name = ft[1].split('_usd')[0]
        # print(f'{crypto_name.upper()} - \tPeriods: {len(new_df)} \tPeriods weight (mean): {crypto_scores_weight:.2f}')
    
    train_set = to_numpy_tuple(train_set)
    val_set = to_numpy_tuple(val_set)

    model = scores_series_model2(train_set)
    opt = setting_optimizer()
    # LOSS FUNCTION: OneHot encoding: categorical_crossentropy,  1d encoding: sparse_categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    train(dirs, model, train_set, val_set)


def load_trained_model(model_dir, model_filename):
    model = scores_series_model2()
    opt = setting_optimizer()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
    
    model.load_weights(os.path.join(model_dir, model_filename))   

    rprint(f'[italic red] Loaded model {model_filename}. [/italic red]')
    return model, HISTORY_POINTS
    # Wczytuje strukturę modelu, wypełnia ją wagami zapisanymi w checkpoincie.


def predict_using_trained_model(model, scores_timeseries):
    ts_df = scores_timeseries.pct_change()  #pierwsza wartość stanie się NaN
    ts_df.dropna(inplace=True)              #usuwa pierwszą wartość (naddatek, więc zostaje HISTORY_POINTS próbek)
    ts_nda = preprocessing.scale(ts_df)  #zwraca numpy.ndarray

    return model.predict(np.array([ts_nda]))[0]  # zwraca np array z 2 liczbami
    #Jako argument podaje się jedną sekwencję skorów. Funkcja zwraca werdykt, czy kupić (lub sprzedać)


#==testowe odpalanie z poziomu tego pliku
def quick_launch_ml():
    dir0 = os.path.dirname(os.path.dirname( __file__ ))
    # dir0 = os.path.join( os.path.dirname( __file__ ), '..' )

    dirt = os.path.join(dir0, 'twitter_scraped')
    scores_path = os.path.join(dirt, 'wybrane_skory')
    selected_files_tuples = [('1643305503scores2h.csv', 'dot_usd.csv'),  #(unfiltered)
                            ('1643306002scores2h.csv', 'xrp_usd.csv'),
                            ('1643309954scores2h.csv', 'eth_usd.csv'),
                            ('1643312609scores2h.csv', 'btc_usd.csv')]

    dirs = {'dir0': dir0,
            'prices': os.path.join(dir0, 'resources', 'prices'),
            'selected_scores': scores_path}

    perform_machine_learning(dirs, selected_files_tuples)
    #Funkcja zawierająca ścieżki folderów, służąca testowemu, szybkiemu odpalania skryptów do trenowania.
    # Tak, by możną ją było wywoływać z tego pliku, a nie z maina.


def test_predicting():
    dir0 = os.path.dirname(os.path.dirname( __file__ ))
    model_dir = os.path.join(dir0, 'machine_learning', 'Price_ML', 'models')
    model_filename = 'PriceLSTM-06ep-0.781accu'
    model, _ = load_trained_model(model_dir, model_filename)

    files_tuple = ('1643305503scores2h.csv', 'dot_usd.csv') #(wytrenowany dot)
    scores_dir = os.path.join(dir0, 'twitter_scraped', 'wybrane_skory')
    scores_fpath = os.path.join(scores_dir, files_tuple[0])
    dirs = {'dir0': dir0,
            'prices': os.path.join(dir0, 'resources', 'prices')}
    # scores_df, _ = du.open_scores(filepath=scores_fpath)
    # score_batches = get_timeseries_batches(scores_df, target_column='count', seq_len=HISTORY_POINTS + 1) #'count' by dać cokolwiek
    # predictions = []
    # for batch, temp_target in score_batches[1:30]:
    #     batch_s = pd.DataFrame(batch).squeeze()     #tworzy pd.Series()
    #     predictions.append(predict_using_trained_model(model, batch_s))
    scores_df, crypto_scores_weight = preprocess_scores(dirs, scores_fpath, files_tuple[1])
    # scores_df = divide_to_equal_parts(scores_df)  # ZAMIENIĆ NA FUNKCJĘ, KTÓRA SAMA ROZDZIELA PO RÓWNO WG KATEGORII
    train_ds = panda_to_numpy_tuple(scores_df)

    predictions = []
    for time_series, target in list(zip(*train_ds)):
        predictions.append([model.predict( np.array([time_series]) )[0], target])
    pdf = pd.DataFrame(predictions)

    print(predictions)

quick_launch_ml()
# test_predicting()






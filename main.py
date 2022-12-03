import pandas as pd
import numpy as np
import keras
import os
import soundfile as sf
import tensorflow as tf
import librosa
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from mixup_layer import MixupLayer
from subcluster_adacos import SCAdaCos
from scipy.stats import hmean
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def predict_on_threshold(predictions: np.ndarray, known_idx: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predict_openset = np.zeros(predictions.shape)
    # only keep largest value
    for j in range(predictions.shape[0]):
        max_value = np.amax(predictions[j, :])
        idx_max_value = np.argmax(predictions[j, :])
        predict_openset[j, idx_max_value] = max_value
    # handle known unknowns
    predict_openset *= known_idx
    # compare to threshold
    for j in range(predictions.shape[0]):
        max_value = np.amax(predict_openset[j, :])
        idx_max_value = np.argmax(predict_openset[j, :])
        if max_value > threshold:
            predict_openset[j, idx_max_value] = 1
    return predict_openset.astype(np.int32)


class MagnitudeSpectrogram(tf.keras.layers.Layer):
    """
    Compute magnitude spectrograms. Taken from:
    https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
    """

    def __init__(self, sample_rate, fft_size, hop_size, f_min=0.0, f_max=None, **kwargs):
        super(MagnitudeSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2

    def build(self, input_shape):
        super(MagnitudeSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=False)
        magnitude_spectrograms = tf.abs(spectrograms)
        magnitude_spectrograms = tf.expand_dims(magnitude_spectrograms, 3)
        return magnitude_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(MagnitudeSpectrogram, self).get_config())
        return config

def mixupLoss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true=y_pred[:, :, 1], y_pred=y_pred[:, :, 0])


def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line / np.math.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


def model_emb_cnn(num_classes, raw_dim, n_subclusters, use_bias=False):
    data_input = tf.keras.layers.Input(shape=(raw_dim, 1), dtype='float32')
    label_input = tf.keras.layers.Input(shape=(num_classes), dtype='float32')
    y = label_input
    x = data_input
    l2_weight_decay = tf.keras.regularizers.l2(1e-5)
    x_mix, y = MixupLayer(prob=1)([x, y])

    # FFT
    x = tf.keras.layers.Lambda(lambda x: tf.math.abs(tf.signal.fft(tf.complex(x[:,:,0], tf.zeros_like(x[:,:,0])))[:,:int(raw_dim/2)]))(x_mix)
    x = tf.keras.layers.Reshape((-1,1))(x)
    x = tf.keras.layers.Conv1D(128, 256, strides=64, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 64, strides=32, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 16, strides=4, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    emb_fft = tf.keras.layers.Dense(128, name='emb_fft', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)

    # magnitude
    x = tf.keras.layers.Reshape((64000,))(x_mix)
    x = MagnitudeSpectrogram(16000, 1024, 512, f_max=8000, f_min=200)(x)
    x = tf.keras.layers.Lambda(lambda x: x - tf.math.reduce_mean(x, axis=1, keepdims=True))(x) # CMN-like normalization
    x = tf.keras.layers.BatchNormalization(axis=-2)(x)

    # first block
    x = tf.keras.layers.Conv2D(16, 7, strides=2, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    # second block
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # third block
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=32, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # fourth block
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=64, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # fifth block
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=128, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=use_bias)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    xr = tf.keras.layers.ReLU()(x)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.ReLU()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=use_bias)(xr)
    x = tf.keras.layers.Add()([x, xr])

    x = tf.keras.layers.MaxPooling2D((10, 1), padding='same')(x)
    x = tf.keras.layers.Flatten(name='flat')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    emb_mel = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, name='emb_mel', use_bias=use_bias)(x)

    # prepare output
    x = tf.keras.layers.Concatenate(axis=-1)([emb_fft, emb_mel])
    output = SCAdaCos(n_classes=num_classes, n_subclusters=n_subclusters, trainable=False)([x, y, label_input])
    loss_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([output, y])

    return data_input, label_input, loss_output


########################################################################################################################
# Load data and compute embeddings
########################################################################################################################
target_sr = 16000 # original sr is 16000
#openness = 'high' # 'high' or 'middle' or 'low'
#shots = 4 # 1 or 2 or 4

# training parameters
epochs = 100
alpha = 1
n_subclusters = 1
iterations = 5

if not os.path.exists('./trained_models'):
   os.makedirs('./trained_models')
for openness in ['low', 'middle', 'high']:
    for shots in [1, 2, 4]:
        info = pd.read_csv('./meta/' + str(shots) + 'shot_openness' + openness + '.csv')

        # load data
        print('Loading data')
        data_raw_path = str(target_sr) + '_data_raw.npy'
        if os.path.isfile(data_raw_path):
            data_raw = np.load(data_raw_path)
        else:
            data_raw = []
            for file_path in tqdm(info['filename']):
                wav, fs = sf.read('./' + file_path)
                raw = librosa.core.to_mono(wav.transpose()).transpose()[:4 * target_sr]
                data_raw.append(raw)
            # reshape array and store
            data_raw = np.expand_dims(np.array(data_raw, dtype=np.float32), axis=-1)
            np.save(data_raw_path, data_raw)
        labels = np.array([info['filename'][k].split('/')[1] for k in np.arange(len(info['filename']))])
        unknown = info['target'] == 'unknown'
        folds = info['fold']
        labels[folds==41] = 'unknown'

        # encode labels
        num_classes = len(np.unique(labels))
        le = LabelEncoder()
        labels_enc = le.fit_transform(labels)
        y_cat = keras.utils.np_utils.to_categorical(labels_enc, num_classes=num_classes)
        total_folds = {1:40, 2:20, 4:10}[shots]
        final_results = np.zeros((iterations, total_folds, 5))
        for fold in tqdm(np.arange(total_folds)):
            train = folds==fold+1
            y_cat_train = y_cat[train]
            y_cat_test = y_cat[~train]
            data_raw_train = data_raw[train]
            data_raw_test = data_raw[~train]
            unknown_train = unknown[train]
            unknown_test =  unknown[~train]
            batch_size = 8*shots#int(2**np.floor(np.log(np.sum(train))/np.log(2)-2))

            scores_test = np.zeros((np.sum(~train), num_classes))
            scores_train = np.zeros((np.sum(train), num_classes))

            for k_iter in np.arange(iterations):
                # compile model
                data_input, label_input, loss_output = model_emb_cnn(num_classes=num_classes, raw_dim=data_raw_train.shape[1],
                                                                     n_subclusters=n_subclusters, use_bias=False)
                model = tf.keras.Model(inputs=[data_input, label_input], outputs=[loss_output])
                model.compile(loss=[mixupLoss], optimizer=tf.keras.optimizers.Adam())
                #print(model.summary())
                # fit model
                weight_path = './trained_models/wts_' + str(k_iter+1) + 'k_' + str(target_sr) + '_' + str(fold) + '_' + openness + '_' + str(shots) + '.h5'
                if not os.path.isfile(weight_path):
                    model.fit(
                        [data_raw_train, y_cat_train], y_cat_train, verbose=0,
                        batch_size=batch_size, epochs=epochs,
                        #validation_data=([data_raw_test, y_cat_test], y_cat_test)
                        )
                    model.save(weight_path)
                else:
                    model = tf.keras.models.load_model(weight_path,
                                                       custom_objects={'MixupLayer': MixupLayer, 'mixupLoss': mixupLoss,
                                                                       'SCAdaCos': SCAdaCos,
                                                                       'MagnitudeSpectrogram': MagnitudeSpectrogram})

                # predict class probabilities
                #test_probs = model.predict([data_raw_test, np.zeros((np.sum(~train), num_classes))], batch_size=batch_size)[:,:,0]
                #plt.imshow(test_probs, aspect='auto')
                #plt.show()

                # extract embeddings
                emb_model = tf.keras.Model(model.input, model.layers[-3].output)
                train_embs = emb_model.predict([data_raw_train, np.zeros((np.sum(train), num_classes))], batch_size=batch_size)
                test_embs = emb_model.predict([data_raw_test, np.zeros((np.sum(~train), num_classes))], batch_size=batch_size)

                # length normalization
                x_train_ln = length_norm(train_embs)
                x_test_ln = length_norm(test_embs)

                # compute cosine distances
                for j_class in np.unique(np.argmax(y_cat_train,axis=1)):
                    class_embs = x_train_ln[np.argmax(y_cat_train,axis=1) == j_class]
                    scores_test[:,j_class] = np.max(np.dot(x_test_ln, class_embs.transpose()), axis=-1)
                    scores_train[:,j_class] = np.max(np.dot(x_train_ln, class_embs.transpose()), axis=-1)

                #plt.imshow(scores_test, aspect='auto')
                #plt.show()

                # compute and print results
                known_idx = np.max(y_cat_train * np.expand_dims(~unknown_train, axis=1), axis=0)
                #plt.plot(np.max(scores_test * known_idx, axis=1))
                #plt.show()
                threshold = 0.6
                # check if no anomalous training samples are available
                if openness == 'high':
                    threshold = 0.8
                pred_test = predict_on_threshold(scores_test, known_idx, threshold=threshold)
                # pred_test = predict_on_threshold(scores_test*known_idx, threshold=0.7)
                # plt.imshow(pred_test, aspect='auto')
                # plt.show()
                acc_kk = accuracy_score(y_cat_test[~unknown_test] * known_idx, pred_test[~unknown_test])
                final_results[k_iter, fold, 0] = acc_kk
                acc_u = accuracy_score(np.zeros(y_cat_test[unknown_test].shape), pred_test[unknown_test])
                if openness == 'low':
                    # acc_ku
                    final_results[k_iter, fold, 1] = acc_u
                elif openness == 'middle':
                    # acc_kuu
                    final_results[k_iter, fold, 2] = acc_u
                elif openness == 'high':
                    # acc_uu
                    final_results[k_iter, fold, 3] = acc_u
                acc_w = 0.5 * acc_kk + 0.5 * acc_u
                final_results[k_iter, fold, 4] = acc_w
                #print(acc_w)
            # print('####################')
            # print(np.mean(final_results, axis=0)[fold])
            # print(np.std(final_results, axis=0)[fold])
            # print('####################')
        print('####################')
        print('final results for ' + openness + ' openness and ' + str(shots) + ' shots:')
        print(np.mean(final_results, axis=(0,1)))
        print(np.std(final_results, axis=(0,1)))

print('####################')
print('>>>> finished! <<<<<')
print('####################')

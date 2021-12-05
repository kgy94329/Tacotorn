import librosa
import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow.keras.layers import RNN, Dense, Dropout, Conv1D, MaxPooling1D, GRUCell, Embedding, GRU, BatchNormalization, Bidirectional, Activation
from util.hparams import HyperParams as hp
import sys
sys.path.append('..')
from text.symbols import get_symbols

#メインモデル(エンコーダー＋デコーダー)
class Tacotron(tf.keras.Model):
    def __init__(self, k, conv_dim):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(k, conv_dim)
        self.decoder = Decoder()
        

    def call(self, enc_input, sequence_length, dec_input, is_training = True):
        batch = dec_input.shape[0]
        x = self.encoder(enc_input, sequence_length, is_training)
        x = self.decoder(batch, dec_input, x)
        return x

class Encoder(tf.keras.Model):
    def __init__(self, k, conv_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(hp.n_symbols, hp.embed_size)
        self.pre_net = Pre_net()
        self.cbhg = CBHG(k, conv_dim)
    
    def call(self, enc_input, sequence_length, is_training):
        input_data = self.embedding(enc_input)
        x = self.pre_net(input_data, is_training)
        encoded = self.cbhg(x, sequence_length, is_training = is_training)
        return encoded #デコーダーの初期状態(隠れ状態)となるベクトルであり、Attention Valueを生成するためのベクトル

#瓶目現象を緩和させるモジュール
class Pre_net(tf.keras.Model):
    def __init__(self):
        super(Pre_net, self).__init__()
        self.dense1 = Dense(256)
        self.dense2 = Dense(128)
    
    def call(self, input_data, is_training):
        x = self.dense1(input_data)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x, training=is_training)
        x = self.dense2(x)
        x = Activation('relu')(x)
        prenet = Dropout(0.5)(x, training = is_training)
        return prenet

#入力されたデータを畳み込み演算を行い、双方向GRU演算を行う層
class CBHG(tf.keras.Model):
    def __init__(self, k, conv_dim):
        super(CBHG, self).__init__()
        self.k = k
        self.conv_dim = conv_dim
        self.conv_bank = []
        for k in range(1, self.k + 1):
            x = Conv1D(128, kernel_size = k, padding = 'same', activation='relu')
            self.conv_bank.append(x)
        self.bn = BatchNormalization()
        self.conv1 = Conv1D(conv_dim[0], kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(conv_dim[1], kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

        self.proj = Dense(128)
        self.dense1 = Dense(128)
        self.dense2 = Dense(128,  bias_initializer=tf.constant_initializer(-1.0))

        self.gru_fw = GRUCell(128)
        self.gru_bw = GRUCell(128)

    def call(self, input_data, sequence_length, is_training):
        x = tf.concat([Activation('relu')(self.bn(self.conv_bank[i](input_data))) for i in range(self.k)], axis=-1)
        x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        highway_input = input_data + x

        if self.k == 8:
            highway_input = self.proj(highway_input)
        
        for _ in range(4):
            highway = self.dense1(highway_input)
            highway = Activation('relu')(highway)
            T = self.dense2(highway_input)
            T = Activation('sigmoid')(T)
            highway_input = highway * T + highway_input * (1.0 - T)
        
        #tf.compay.v1.nn.bidirectional_dynamic_rnnはなくなる可能性があるので後で修正が必要。
        x, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            self.gru_fw, 
            self.gru_bw,
            highway_input,
            sequence_length=sequence_length,
            dtype=tf.float32)
        x = tf.concat(x, axis=2)
        return x

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pre_net = Pre_net()
        self.attention_rnn = GRU(hp.decoder_dim, return_sequences=True)
        self.proj1 = Dense(hp.decoder_dim) #256
        self.dec_rnn1 = GRU(hp.decoder_dim, return_sequences=True)
        self.dec_rnn2 = GRU(hp.decoder_dim, return_sequences=True)
        self.proj2 = Dense(hp.mel_dim * hp.reduction) #80 * 5
    
    def call(self, batch, dec_input, enc_output):
        x = self.pre_net(dec_input, is_training=True)
        x = self.attention_rnn(x)
        context, alignment = Attention(x, enc_output)

        dec_rnn_input = self.proj1(context)
        dec_rnn_input += self.dec_rnn1(dec_rnn_input)
        dec_rnn_input += self.dec_rnn2(dec_rnn_input)

        dec_out = self.proj2(dec_rnn_input)
        mel_out = tf.reshape(dec_out, [batch, -1, hp.mel_dim])

        return mel_out, alignment


class Post_net(tf.keras.Model):
    def __init__(self, k, conv_dim):
        super(Post_net, self).__init__()
        self.cbhg = CBHG(k, conv_dim)
        self.dense = Dense(hp.n_fft // 2 + 1)
    
    def call(self, mel_input, is_training):
        x = self.cbhg(mel_input, None, is_training=is_training)
        x = self.dense(x)
        return x

def Attention(query, value):
    alignment = tf.nn.softmax(tf.matmul(query, value, transpose_b=True))
    #行列の掛け算(Attention Valueでありcontext vectorとも呼びます。)
    context = tf.matmul(alignment, value)
    #2つの行列を連結させる(Attetion Valueと隠れ状態ベクトルを結合する。)
    context = tf.concat([context, query], axis = -1)
    #転置行列
    alignment = tf.transpose(alignment, [0, 2, 1])
    return context, alignment

def griffin_lim(spectrogram):
    spec = deepcopy(spectrogram)
    for i in range(50):
        est_wav = librosa.istft(spec, hop_length=hp.hop_length, win_length=hp.win_length, window='hann')
        est_stft = librosa.stft(est_wav, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spec = spectrogram * phase
    wav = librosa.istft(spec, hop_length=hp.hop_length, win_length=hp.win_length, window='hann')
    return np.real(wav)

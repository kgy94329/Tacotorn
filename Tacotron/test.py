import os
import glob
import random
import traceback
import scipy
import scipy.io.wavfile as sio
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from jamo import hangul_to_jamo
from text import sequence_to_text
from tqdm import tqdm
from model.tacotron import Tacotron, Post_net, griffin_lim
from tensorflow.keras.optimizers import Adam
from util.hparams import HyperParams as hp
from util.plot_alignment import plot_alignment
from text import sequence_to_text, text_to_sequence, _clean_text

sentences = [
  "어서오세요 권용일씨 환영합니다.", "배고프니 빨리 앉으셨으면 좋겠어요"
]

checkpoint_dir = hp.check_point_dir + hp.pre_cp
post_checkpoint_dir = hp.check_point_dir + hp.post_cp
save_dir = hp.out_dir
os.makedirs(save_dir, exist_ok=True)


def test_step(text, idx):
    seq = text_to_sequence(text, hp.text_cleaners)
    enc_input = np.asarray([seq], dtype=np.int32)
    sequence_length = np.asarray([len(seq)], dtype=np.int32)
    print(idx)
    dec_input = np.zeros((enc_input.shape[0], hp.max_iter, hp.mel_dim), dtype=np.float32)

    pred = []
    for i in tqdm(range(1, hp.max_iter+1)):
        mel_out, alignment = model(enc_input, sequence_length, dec_input, is_training=False)
        if i < hp.max_iter:
            dec_input[:, i, :] = mel_out[:, hp.reduction * i - 1, :]
        pred.extend(mel_out[:, hp.reduction * (i-1) : hp.reduction * i, :])

    pred = np.reshape(np.asarray(pred), [-1, hp.mel_dim])
    alignment = np.squeeze(alignment, axis=0)

    np.save(os.path.join(save_dir, 'mel-{}'.format(idx)), pred, allow_pickle=False)

    input_seq = sequence_to_text(seq)
    alignment_dir = os.path.join(save_dir, 'align-{}.png'.format(idx))
    plot_alignment(alignment, alignment_dir, input_seq)

def post_test_step(mel, idx):
    mel = np.expand_dims(mel, axis=0)
    pred = post_model(mel, is_training=False)

    pred = np.squeeze(pred, axis=0)
    pred = np.transpose(pred)

    #spectrogram2wav
    pred = (np.clip(pred, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    pred = np.power(10.0, pred * 0.05)
    wav = griffin_lim(pred)
    wav = scipy.signal.lfilter([1], [1, -hp.preemphasis], wav)
    wav, _ = librosa.effects.trim(wav)
    endpoint = librosa.effects.split(wav, frame_length=hp.win_length, hop_length=hp.hop_length)[0, 1]
    wav = wav[:endpoint]
    wav = wav.astype(np.float32)
    sio.write(os.path.join(save_dir, '{}.wav'.format(idx)), hp.sr, wav)


model = Tacotron(k=16, conv_dim=[128, 128])
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

post_model = Post_net(k=8, conv_dim=[256, hp.mel_dim])
post_optimizer = Adam()
post_step = tf.Variable(0)
post_checkpoint = tf.train.Checkpoint(optimizer=post_optimizer, model=post_model, step=post_step)
post_checkpoint.restore(tf.train.latest_checkpoint(post_checkpoint_dir)).expect_partial()

def synthesize():    
    print('text process')
    for i, text in enumerate(sentences):
        test_step(text, i)
    
    mel_list = glob.glob(os.path.join(save_dir, '*.npy'))
    
    print('make wav')
    for i, fn in enumerate(mel_list):
        print('{}번째 텍스트를 합성하고 있습니다.'.format(i))
        mel = np.load(fn)
        post_test_step(mel, i)
       


if __name__ == '__main__':
    synthesize()
    print('Done')
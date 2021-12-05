import os
import re
import librosa
import codecs
import numpy as np
import unicodedata
from unidecode import unidecode
from tqdm import tqdm
from util.hparams import HyperParams as hp
from text import text_to_sequence


def load_data():
    # Load vocabulary
    fpaths, texts = [], []
    transcript = os.path.join(hp.data, hp.script)
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in lines:
        metadata = line.strip().split("|")
        fname = metadata[0]
        text = metadata[1]
        fpath = os.path.join(hp.data, "wavs", fname + ".wav")
        fpaths.append(fpath)
        texts.append((fname, text))
        
    return fpaths, texts

def text_prepro(texts):
    text_lengths = []
    for fname, text in tqdm(texts):
        text = text_to_sequence(text, hp.text_cleaners)
        text_name = '{}_text.npy'.format(fname)
        text_lengths.append(len(text))
        np.save(os.path.join(out_dir + '/text', text_name), text, allow_pickle=False)
    return text_lengths


def audio_prepro(fpath):
    fname = os.path.basename(fpath)

    wav, sr = librosa.load(fpath, sr=hp.sr)

    wav, _ = librosa.effects.trim(wav)

    # Preemphasis
    wav = np.append(wav[0], wav[1:] - hp.preemphasis * wav[:-1])

    # stft
    linear = librosa.stft(y = wav,
                        n_fft=hp.n_fft,
                        hop_length=hp.hop_length,
                        win_length=hp.win_length)
        
    # magnitude spectrogram (진도(지진의 규모, 별의 광도) 스펙트로그램)
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram (Melody spectrogram)
    #mel-filter bank를 생성
    mel_filter = librosa.filters.mel(hp.sr, hp.n_fft, hp.mel_dim)  # (mel_dim, 1+n_fft//2)
    mel = np.dot(mel_filter, mag)  # (n_mels, t) = (80, T)

    # to decibel 데시벨화
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return fname, mel, mag

if __name__ == '__main__':
    out_dir = hp.data_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/text', exist_ok=True)
    os.makedirs(out_dir + '/mels', exist_ok=True)
    os.makedirs(out_dir + '/mags', exist_ok=True)
    os.makedirs(out_dir + '/dec', exist_ok=True)
    
    fpaths, texts = load_data()
    # text
    print('Load Text')
    text_lengths = text_prepro(texts)
    np.save(os.path.join(out_dir + '/text_len.npy'), np.array(text_lengths))
    print('Text Done')
    
    print('audio prepro')
    
    mel_len_list = []
    for idx, fpath in enumerate(tqdm(fpaths)):
        fname, mel, mag = audio_prepro(fpath)
        
        mel_len_list.append([mel.shape[0], idx])
        
        remainder = mel.shape[0] % hp.reduction
        if remainder != 0:
            mel = np.pad(mel, [[0, hp.reduction - remainder], [0, 0]], mode = 'constant')
            mag = np.pad(mag, [[0, hp.reduction - remainder], [0, 0]], mode = 'constant')


        np.save(hp.data_dir + "/mels/{}".format(fname.replace("wav", "npy")), mel, allow_pickle=False)
        np.save(hp.data_dir + "/mags/{}".format(fname.replace("wav", "npy")), mag, allow_pickle=False)
        
        mel = mel.reshape((-1, hp.mel_dim * hp.reduction))
        dec_input = np.concatenate((np.zeros_like(mel[:1, :]), mel[:-1, :]), axis=0)
        dec_input = dec_input[:, -hp.mel_dim:]
        
        np.save(hp.data_dir + "/dec/{}".format(fname.replace("wav", "npy")), dec_input, allow_pickle=False)
    
    mel_len = sorted(mel_len_list)
    np.save(os.path.join(out_dir + '/mel_len.npy'), np.array(mel_len))
    
    print('audio done')


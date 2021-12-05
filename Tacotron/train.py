import os
import glob
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MAE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tacotron import Tacotron, Post_net
from util.hparams import HyperParams as hp
from util.plot_alignment import plot_alignment
from text import sequence_to_text
from tqdm import tqdm

print('시작')
data_dir = hp.data_dir
text_list = glob.glob(os.path.join(data_dir + '/text', '*.npy'))
mel_list = glob.glob(os.path.join(data_dir + '/mels', '*.npy'))
mag_list = glob.glob(os.path.join(data_dir + '/mags', '*.npy'))
dec_list = glob.glob(os.path.join(data_dir + '/dec', '*.npy'))
fn = os.path.join(data_dir, 'mel_len.npy')
if not os.path.isfile(fn):
    mel_len_list = []
    for i in tqdm(range(len(mel_list))):
        mel_length = np.load(mel_list[i]).shape[0]
        mel_len_list.append([mel_length, i])
    mel_len = sorted(mel_len_list)
    np.save(os.path.join(data_dir + '/mel_len.npy'), np.array(mel_len))

text_len = np.load(os.path.join(data_dir + '/text_len.npy'))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))
print('1차작업')
def Datagenerator():
    while True:
        idx_list = np.random.choice(len(mel_list), hp.batch_size * hp.batch_size, replace=False)
        idx_list = sorted(idx_list)
        idx_list = [idx_list[i : i + hp.batch_size] for i in range(0, len(idx_list), hp.batch_size)]
        random.shuffle(idx_list)

        for idx in idx_list:
            random.shuffle(idx)

            text = [np.load(text_list[mel_len[i][1]]) for i in idx]
            mel = [np.load(mel_list[mel_len[i][1]]) for i in idx]
            mag = [np.load(mag_list[mel_len[i][1]]) for i in idx]
            dec = [np.load(dec_list[mel_len[i][1]]) for i in idx]
            text_length = [text_len[mel_len[i][1]] for i in idx]

            text = pad_sequences(text, padding='post')
            dec = pad_sequences(dec, padding='post', dtype='float32')
            mel = pad_sequences(mel, padding='post', dtype='float32')
            mag = pad_sequences(mag, padding='post', dtype='float32')
            yield(text, dec, mel, mag, text_length)

@tf.function(experimental_relax_shapes = True)
def train_step(model, enc_input, dec_input, dec_target, text_length):
    with tf.GradientTape() as tape:
        # print('text_length', text_length)
        pred, alignment = model(enc_input, text_length, dec_input, is_training = True)
        # print(pred.shape)
        # print(dec_target.shape)
        loss = tf.reduce_mean(MAE(dec_target, pred))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, pred[0], alignment[0]

@tf.function(experimental_relax_shapes=True)
def post_train_step(model, mel_input, mag_target):
    with tf.GradientTape() as tape:
        pred = model(mel_input, is_training=True)
        loss = tf.reduce_mean(MAE(mag_target, pred))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, pred[0]


dataset = tf.data.Dataset.from_generator(Datagenerator,
    output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
    output_shapes = (tf.TensorShape([hp.batch_size, None]), #text
                    tf.TensorShape([hp.batch_size, None, hp.mel_dim]), #dec
                    tf.TensorShape([hp.batch_size, None, hp.mel_dim]), #mel
                    tf.TensorShape([hp.batch_size, None, hp.n_fft // 2 + 1]), #mag
                    tf.TensorShape([hp.batch_size])
                    )
                )
dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(dataset)
model = Tacotron(k = 16, conv_dim = [128, 128])
post_model = Post_net(k = 8, conv_dim=[256, hp.mel_dim])
optimizer = Adam()
step = tf.Variable(0)
post_step = tf.Variable(0)

checkpoint_dir = hp.check_point_dir + hp.pre_cp
post_checkpoint_dir = hp.check_point_dir + hp.post_cp
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(post_checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model = model, step = step)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
post_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model = post_model, step = post_step)
post_manager = tf.train.CheckpointManager(post_checkpoint, post_checkpoint_dir, max_to_keep=5)

checkpoint.restore(manager.latest_checkpoint)
post_checkpoint.restore(post_manager.latest_checkpoint)
print('2차작업')

if manager.latest_checkpoint:
    print('-'*40)
    print('Restore checkpoint from {}'.format(manager.latest_checkpoint))
    print('-'*40)

if post_manager.latest_checkpoint:
    print('-'*40)
    print('Restore post_checkpoint from {}'.format(post_manager.latest_checkpoint))
    print('-'*40)

try:
    with tf.device('/device:GPU:0'):
        for text, dec, mel, mag, text_length in dataset:
            loss, pred, alignment = train_step(model, text, dec, mel, text_length)
            checkpoint.step.assign_add(1)
            print('step: {}, Loss: {:.5f}'.format(int(checkpoint.step), loss))

            post_loss, post_pred = post_train_step(post_model, mel, mag)
            post_checkpoint.step.assign_add(1)
            print("post_Step: {}, post_Loss: {:.5f}".format(int(post_checkpoint.step), post_loss))

            if int(checkpoint.step) % 500 == 0:
                checkpoint.save(file_prefix = os.path.join(checkpoint_dir, 'step-{}'.format(int(checkpoint.step))))
                post_checkpoint.save(file_prefix=os.path.join(post_checkpoint_dir, 'step-{}'.format(int(post_checkpoint.step))))

                input_seq = sequence_to_text(text[0].numpy())
                input_seq = input_seq[:text_length[0].numpy()]
                alignment_dir = os.path.join(checkpoint_dir, 'step-{}-align.png'.format(int(checkpoint.step)))
                plot_alignment(alignment, alignment_dir, input_seq)

        

except Exception:
    traceback.print_exc()
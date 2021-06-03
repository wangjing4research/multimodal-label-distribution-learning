import numpy as np
import pandas as pd

from evaluate import cal_loss
import EF_MLDL

from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

import tensorflow as tf
from tensorflow.keras import backend as K

def one_hot_labels(labels):
	labels -= min(labels)
	size = int(max(labels))
	labels_onehot = []
	for i in labels:
		temp = np.zeros(size+1)
		temp[int(i)] = 1
		labels_onehot.append(temp)
	return np.array(labels_onehot)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

if __name__ == '__main__':
	train_text_env = np.load('./cmumosi/train_text_env.npy')
	train_visual_env = np.load('./cmumosi/train_visual_env.npy')
	train_audio_env = np.load('./cmumosi/train_audio_env.npy')
	train_text = np.load('./cmumosi/train_text.npy')
	train_visual = np.load('./cmumosi/train_visual.npy')
	train_audio = np.load('./cmumosi/train_audio.npy')
	train_labels = np.load('./cmumosi/train_labels.npy')
	train_labels = one_hot_labels(train_labels)

	val_text_env = np.load('./cmumosi/val_text_env.npy')
	val_visual_env = np.load('./cmumosi/val_visual_env.npy')
	val_audio_env = np.load('./cmumosi/val_audio_env.npy')
	val_text = np.load('./cmumosi/val_text.npy')
	val_visual = np.load('./cmumosi/val_visual.npy')
	val_audio = np.load('./cmumosi/val_audio.npy')
	val_labels = np.load('./cmumosi/val_labels.npy')
	val_labels = one_hot_labels(val_labels)

	test_text_env = np.load('./cmumosi/test_text_env.npy')
	test_visual_env = np.load('./cmumosi/test_visual_env.npy')
	test_audio_env = np.load('./cmumosi/test_audio_env.npy')
	test_text = np.load('./cmumosi/test_text.npy')
	test_visual = np.load('./cmumosi/test_visual.npy')
	test_audio = np.load('./cmumosi/test_audio.npy')
	test_labels = np.load('./cmumosi/test_labels.npy')
	test_labels = one_hot_labels(test_labels)

	train = [train_text_env,train_visual_env,train_audio_env, train_text,train_visual,train_audio]
	val = [val_text_env, val_visual_env, val_audio_env, val_text, val_visual, val_audio]
	test = [test_text_env, test_visual_env, test_audio_env, test_text, test_visual, test_audio]

	filepath = "./models/ef_mldl.h5"
	checkpoint = ModelCheckpoint(
		filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
	earlystopping = EarlyStopping(
		monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='max')

	callbacks = [checkpoint, earlystopping]

	learning_rate = CustomSchedule(256)
	custom_adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

	model = EF_MLDL(train_text.shape[1], train_text.shape[2], train_visual.shape[2], train_audio.shape[2], train_labels.shape[1])
	model.compile(loss='categorical_crossentropy', optimizer=custom_adam, metrics=['accuracy'])
	result = model.fit(train, train_labels, batch_size=128, epochs=100, verbose=1,validation_data=[val, val_labels],callbacks=callbacks,shuffle=True)

	model.load_weights("./models/ef_mldl.h5")
	pre = model.predict(test, batch_size=128)
	cal_loss(test_labels, pre)
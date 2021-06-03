import mmsdk
import os
import re
import numpy as np
import pandas as pd
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

def avg_collapse(intervals: np.array, features: np.array) -> np.array:
	try:
		return np.average(features, axis=0)
	except:
		return features

def max_collapse(intervals: np.array, features: np.array) -> np.array:
	try:
		return np.max(features, axis=0)
	except:
		return features

def split_dataset(dataset, train_split, val_split, test_split):
	# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
	EPS = 0

	# place holders for the final train/dev/test dataset
	train = []
	val = []
	test = []

	# define a regular expression to extract the video ID out of the keys
	pattern = re.compile('(.*)\[.*\]')
	num_drop = 0 # a counter to count how many data points went into some processing issues

	for segment in dataset[label_field].keys():
		# get the video ID and the features out of the aligned dataset
		vid = re.search(pattern, segment).group(1)
		label = dataset[label_field][segment]['features']
		_words = dataset[text_field][segment]['features']
		_visual = dataset[visual_field][segment]['features']
		_acoustic = dataset[acoustic_field][segment]['features']

		# if the sequences are not same length after alignment, there must be some problem with some modalities
		# we should drop it or inspect the data again
		if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
			print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
			num_drop += 1
		continue

		# remove nan values
		label = np.nan_to_num(label)
		words = np.nan_to_num(_words)
		visual = np.nan_to_num(_visual)
		acoustic = np.nan_to_num(_acoustic)

		# z-normalization per instance and remove nan/infs
		visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
		acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

		if vid in train_split:
			train.append(((words, visual, acoustic), label, segment))
		elif vid in val_split:
			val.append(((words, visual, acoustic), label, segment))
		elif vid in test_split:
			test.append(((words, visual, acoustic), label, segment))
		else:
			print(f"Found video that doesn't belong to any splits: {vid}")

	print(f"Total number of {num_drop} datapoints have been dropped.")
	return train,val,test

def add_zeros(data, max_len):
	if data.shape[0]>max_len:
		data = data[:max_len,:]
	elif data.shape[0]<max_len:
		pad_len = max_len-data.shape[0]
		padding = np.zeros((pad_len, data.shape[1]))
		data = np.concatenate([data, padding], axis=0)

	return data

def pad_sequence(data, max_len):
	text_features = []
	visual_features = []
	audio_features = []
	labels = []
	names = []

	for i in range(len(data)):
		features = data[i][0]

		temp = features[0]
		text_features.append(add_zeros(temp, max_len))

		temp = features[1]
		visual_features.append(add_zeros(temp, max_len))

		temp = features[2]
		audio_features.append(add_zeros(temp, max_len))

		label = np.round(data[i][1][0][0])
		labels.append(label)

		name = data[i][2]
		names.append(name)

	return np.array(text_features), np.array(visual_features), np.array(audio_features), np.array(labels), names

if __name__ == '__main__':
	data = sys.argv[1]
	env_flag = sys.argv[2] #0 for multimodal features, else for side-information
	DATA_PATH = sys.argv[3]

	if data == 'mosi':
		DATASET = md.cmu_mosi
	elif data == 'mosei':
		DATASET = md.cmu_mosei
	else:
		print('Sorry for not supported.')
	
	# download data
	dataset = md.mmdataset(DATASET.highlevel, DATA_PATH)

	# align to words
	if not env_flag:
		dataset.align('glove_vectors', collapse_functions=[avg_collapse]) # multimodal
	else:
		dataset.align('glove_vectors', collapse_functions=[max_collapse]) # side-infomation

	# download labels
	dataset.add_computational_sequences(DATASET.labels,DATA_PATH)
	dataset.align('Opinion Segment Labels')

	# Split train set, valid set, and test set
	train_split = DATASET.standard_folds.standard_train_fold
	val_split = DATASET.standard_folds.standard_valid_fold
	test_split = DATASET.standard_folds.standard_test_fold

	train,val,test = split_dataset(dataset, train_split, val_split, test_split)

	train_text, train_visual, train_audio, train_labels,_ = pad_sequence(train, 20)
	val_text, val_visual, val_audio, val_labels,_ = pad_sequence(val, 20)
	test_text, test_visual, test_audio, test_labels,_ = pad_sequence(test, 20)

	if not env_flag:
		np.save(os.path.join(DATA_PATH, 'train_text.npy'), train_text)
		np.save(os.path.join(DATA_PATH, 'train_visual.npy'), train_visual)
		np.save(os.path.join(DATA_PATH, 'train_audio.npy'), train_audio)
		np.save(os.path.join(DATA_PATH, 'train_labels.npy'), train_labels)

		np.save(os.path.join(DATA_PATH, 'val_text.npy'), val_text)
		np.save(os.path.join(DATA_PATH, 'val_visual.npy'), val_visual)
		np.save(os.path.join(DATA_PATH, 'val_audio.npy'), val_audio)
		np.save(os.path.join(DATA_PATH, 'val_labels.npy'), val_labels)

		np.save(os.path.join(DATA_PATH, 'test_text.npy'), test_text)
		np.save(os.path.join(DATA_PATH, 'test_visual.npy'), test_visual)
		np.save(os.path.join(DATA_PATH, 'test_audio.npy'), test_audio)
		np.save(os.path.join(DATA_PATH, 'test_labels.npy'), test_labels)
	else:
		np.save(os.path.join(DATA_PATH, 'train_text_env.npy'), train_text)
		np.save(os.path.join(DATA_PATH, 'train_visual_env.npy'), train_visual)
		np.save(os.path.join(DATA_PATH, 'train_audio_env.npy'), train_audio)
		np.save(os.path.join(DATA_PATH, 'train_labels_env.npy'), train_labels)

		np.save(os.path.join(DATA_PATH, 'val_text_env.npy'), val_text)
		np.save(os.path.join(DATA_PATH, 'val_visual_env.npy'), val_visual)
		np.save(os.path.join(DATA_PATH, 'val_audio_env.npy'), val_audio)
		np.save(os.path.join(DATA_PATH, 'val_labels_env.npy'), val_labels)

		np.save(os.path.join(DATA_PATH, 'test_text_env.npy'), test_text)
		np.save(os.path.join(DATA_PATH, 'test_visual_env.npy'), test_visual)
		np.save(os.path.join(DATA_PATH, 'test_audio_env.npy'), test_audio)
		np.save(os.path.join(DATA_PATH, 'test_labels_env.npy'), test_labels)
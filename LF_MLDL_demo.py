import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from sklearn.model_selection import train_test_split

from evaluate import cal_loss
import LF_MLDL

def load_data(data, fields, win_size, embed_size, iteration):
	embed_list = []
	data_list = []
	seq_list = []
	for col in fields:
		print(col)
		file_name = 'embed/%s_%d_%d_%d_embed.npy'%(col, win_size, embed_size, iteration)
		embed = np.load(file_name)
		embed_list.append(embed)
		temp = data[col].values
		temp = pad_sequences(temp, value=0)
		data_list.append(temp)
		seq_list.append(temp.shape[1])
	return embed_list, data_list, seq_list

def one_hot_labels(ad_list):
	temp = ad_list['LV1'].values
	x =  [ j for i in temp for j in i]
	labels_num = len(set(x))

	labels = []

	for i in temp:
		y = np.zeros(labels_num,)
		for j in i:
			y[int(j)-1] = 1
		labels.append(y)

	return np.array(labels)

if __name__ == '__main__':
	BASIC_INFO = [ 'AYEAR', 'AMONTH',
		'ADAY', 'AHOUR', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE',
		'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'AGE']

	MULTIMODAL = ['CHARTEVENTS', 'LABEVENTS', 'CPTEVENTS', 'DATETIMEEVENTS',
    	'INPUTEVENTS_CV', 'INPUTEVENTS_MV','PROCEDUREEVENTS_MV','MICROBIOLOGYEVENTS']

	ad_list = pd.read_pickle('MIMIC_WHOLE.pkl')

	basic_embed, basic_data, basic_seq_len = load_data(ad_list, BASIC_INFO, win_size=5, embed_size=32, iteration=20)
	basic_seq = basic_seq_len[0] # basic fields share the same seq len
	mm_embed, mm_data, mm_seq_len = load_data(ad_list, MULTIMODAL, win_size=30, embed_size=128, iteration=20)
	icd_embed, icd_data, icd_seq = load_data(ad_list, ['ICD9_CODE_DOCS'], win_size=50, embed_size=128, iteration=20)

	labels = one_hot_labels(ad_list)
	output_size = labels.shape[1]

	data = basic_data + mm_data + icd_data
	n_samples = len(data[0])
	indices = np.arange(n_samples)
	x_train, x_test, y_train, y_test, x_idx, y_idx = train_test_split(data[0], labels, indices, test_size=0.2, random_state=42)

	train_data = [i[x_idx] for i in data]
	test_data = [i[y_idx] for i in data]

	filepath = "./models/lf_mldl.h5"
	checkpoint = ModelCheckpoint(
		filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
	earlystopping = EarlyStopping(
		monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='max')

	callbacks = [checkpoint, earlystopping]

	model = LF_MLDL(basic_embed, mm_embed, icd_embed[0], basic_seq, mm_seq_len, icd_seq[0], output_size)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	result = model.fit(train_data,y_train, batch_size=128, epochs=100, verbose=1,validation_split=0.1,callbacks=callbacks,shuffle=True)

	model.load_weights('./models/lf_mldl.h5')	
	pre = model.predict(test_data, batch_size=128)
	cal_loss(y_test, pre)
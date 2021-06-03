import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence

import os
import logging
import gensim
from gensim.models import FastText, Word2Vec
from tqdm.autonotebook import *
from sklearn.externals import joblib 


def select_admission(admission, BASIC_INFO):
	ATIME = pd.to_datetime(admission['ADMITTIME'])
	ayear = ATIME.dt.year
	amonth = ATIME.dt.month
	aday = ATIME.dt.day
	ahour = ATIME.dt.hour

	admission['AYEAR'] = ayear
	admission['AMONTH'] = amonth
	admission['ADAY'] = aday
	admission['AHOUR'] = ahour

	ad = pd.DataFrame()
	for col in BASIC_INFO:
		ad[col] = admission[col].copy()
	ad.fillna('nan',inplace=True)

	sparse_col = BASIC_INFO[6:]
	for col in sparse_col:
		lbe = LabelEncoder()
		ad[col] = list(lbe.fit_transform(ad[col].values))

	return ad

def cal_age(ad, patients):
	birth = pd.DataFrame({'SUBJECT_ID':patients['SUBJECT_ID'], 'DOB':patients['DOB']})
	ad = pd.merge(ad,birth, how='left', on='SUBJECT_ID')
	ayear=  ad['AYEAR'].values
	btime = pd.to_datetime(ad['DOB'])
	age = ayear-btime.dt.year
	ad['AGE'] = age
	return ad

def diag_ccs(diagnose, ccs):
	ccs = ccs.applymap(lambda x: x.replace('\'', ''))
	ccs2 = pd.DataFrame({'ICD9_CODE':ccs['\'ICD-9-CM CODE\''], 'LV1':ccs['\'CCS LVL 1\''], 'LV2':ccs['\'CCS LVL 2\''], 'LV3':ccs['\'CCS LVL 3\''], 'LV4':ccs['\'CCS LVL 4\''],})
	diagnose = pd.merge(diagnose,ccs2,how='left',on='ICD9_CODE')
	diagnose[diagnose==' '] = np.NaN
	return diagnose

def group_by(data, col, key):
	data[col] = data[col].astype(str)
	col_list = data.groupby(key)[col].apply(list).reset_index()
	return col_list

def add_labels(ad, diag):
	diag_drop = diag[pd.notnull(diag['LV1'])]
	# add lv1(label) to ad
	lv1_list = group_by(diag_drop, 'LV1', 'HADM_ID')
	lv1_list['LV1'] = lv1_list['LV1'].apply(lambda x: list(set(x)))

	ad = pd.merge(ad, lv1_list, how='left', on='HADM_ID')

	return ad

def set_sort(x):
    x2 = list(set(x))
    x2.sort(key=x.index)
    return x2

def add_mm(ad, multimodal, mm_names):
	for i in range(len(multimodal)):
		modal = multimodal[i]
		col = mm_names[i]
		filename = './data_raw/'+modal+'.csv'
		#print(filename,col)
		data = pd.read_csv(filename)
		data.sort_values('HADM_ID', inplace=True)

		data[col] = data[col].astype(str)
		data_list =  data.groupby(['HADM_ID'])[col].apply(list).reset_index()
		data_list.rename(columns={col:modal},inplace=True)

		temp = data_list[modal].values
		temp = list(map(list,map(set_sort,temp)))
		data_list[modal] = temp

		ad = pd.merge(ad,data_list,how='left',on='HADM_ID')

	return ad

def add_mm2(ad, modal):
	filename = './data_raw/'+modal+'.csv'

	data = pd.read_csv(filename)
	data.sort_values('HADM_ID',inplace=True)
	data.fillna(0, inplace=True)

	temp1 = data['SPEC_ITEMID'].values
	temp2 = data['ORG_ITEMID'].values
	temp3 = []
	for i in range(len(data)):
		x1 = []
		if temp1[i]!=0:
			x1.append(str(int(temp1[i])))
		if temp2[i]!=0:
			x1.append(str(int(temp2[i])))
		temp3.append(x1)

	data[modal] = temp3
	micro = data.groupby(['HADM_ID'])[modal].apply(list).reset_index()
	micro[modal] = micro[modal].apply(lambda x: [j for i in x for j in i])

	temp = micro[modal].values
	temp = list(map(list,map(set_sort,temp)))
	micro[modal] = temp

	ad = pd.merge(ad,micro, how='left',on='HADM_ID' )

	return ad

def group_patients(ad, col_sets, key):
	ad_list = group_by(ad, col_sets[0], key)

	for col in col_sets[1:]:
		temp = group_by(ad, col, key)
		ad_list = pd.merge(ad_list, temp, how='left', on=key)

	return ad_list

def group_mm_res(ad_list, ad, multimodal):
	last = ad.drop_duplicates(subset='SUBJECT_ID',keep='last')

	last2 = pd.DataFrame({'SUBJECT_ID':last['SUBJECT_ID'].copy()})
	for col in multimodal:
		last2[col] = last[col].copy()

	ad_list = pd.merge(ad_list, last2, how='left', on='SUBJECT_ID')

	return ad_list

def add_labels_res(ad_list, ad):
	last = ad.drop_duplicates(subset='SUBJECT_ID',keep='last')
	last2 = pd.DataFrame({'SUBJECT_ID':last['SUBJECT_ID'].copy(), 'LV1':last['LV1'].copy()})
	ad_list = pd.merge(ad_list, last2, how='left', on='SUBJECT_ID')
	ad_list  = ad_list[pd.notnull(ad_list['LV1'])]
	return ad_list

def add_patients(ad_list, patients):
	DOB = pd.to_datetime(patients['DOB'])
	byear = DOB.dt.year
	bmonth = DOB.dt.month
	bday = DOB.dt.day

	patients2 = pd.DataFrame()
	patients2['SUBJECT_ID']=patients['SUBJECT_ID'].copy()
	patients2['GENDER']=patients['GENDER'].copy()
	patients2['BYEAR'] = byear
	patients2['BMONTH'] = bmonth
	patients2['BDAY'] = bday

	ad_list = pd.merge(patients2,ad_list, how='left',on='SUBJECT_ID')
	
	return ad_list

def add_medical_history(ad_list, ad):
	last = ad.drop_duplicates(subset='SUBJECT_ID',keep='last')
	ad_pre = ad.drop(last['HADM_ID'].index)

	ad_pre['ICD9_CODE_DOCS'] = ad_pre['ICD9_CODE'].apply(lambda x: ';'.join(x))
	icd9_list = ad_pre.groupby(['SUBJECT_ID'])['ICD9_CODE_DOCS'].apply(list).reset_index()
	ad_list = pd.merge(ad_list, icd9_list,how='left',on='SUBJECT_ID')

	return ad_list

def get_embedding(data_list, name, win=10, embed_size=128, iteration=30, split_char=';' ):
    docs = []
    for data in data_list:
        docs += list(data)
    
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
    tokenizer.fit_on_texts(docs)
    #token_file_name = 'embed/%s_token.pkl'%(name)
    #joblib.dump(tokenizer,token_file_name)  #模型保存
    
    docs_encode = []
    for data in data_list:
        docs_emb = tokenizer.texts_to_sequences(data.values)
        docs_encode.append(docs_emb)
        #docs_emb = pad_sequences(docs_emb, maxlen=seq_len, value=0)

    index_emb = tokenizer.word_index
    #index_file_name = 'embed/%s_index.npy'%(name)
    #np.save(index_file_name,index_emb)

    input_docs = []
    for i in docs:
        input_docs.append(i.split(split_char))
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=win, seed=2020, workers=4, min_count=1, iter=iteration)
    #file_name = 'embed/%s_%d_%d_%d.txt'%(name, seq_len, embed_size, iteration)
    #w2v.wv.save_word2vec_format(file_name)
    #print("w2v model done")

    #embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False)
    nb_words = len(index_emb)+1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in tqdm(index_emb.items()):
        if i >= nb_words:
            continue
        try:
            embedding_vector = w2v[word]
        except:
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
    print("null cnt",count)
    embed_file_name = 'embed/%s_%d_%d_%d_embed.npy'%(name, win, embed_size, iteration)
    np.save(embed_file_name, embedding_matrix)

    return docs_encode

def feild_embed(data_list, feilds, win_size=5, embed_size=128, iter_size=20):
	for col in feilds:
		docs = []
		for data in data_list:
			temp = data[col].apply(lambda x: ';'.join(x))
			docs.append(temp)

		docs_encode = get_embedding(docs, col, win=win_size, embed_size=embed_size, iteration=iter_size)
		
		for i in range(len(data_list)):
			data_list[i][col] = docs_encode[i]

	return data_list

if __name__ == '__main__':
	DATA_PATH = sys.argv[1]
	SAVE_PATH = sys.argv[2]

	BASIC_INFO = ['SUBJECT_ID', 'HADM_ID', 'AYEAR', 'AMONTH',
		'ADAY', 'AHOUR', 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
		'INSURANCE', 'LANGUAGE', 'RELIGION',
		'MARITAL_STATUS', 'ETHNICITY']

	MULTIMODAL = [ 'CHARTEVENTS', 'LABEVENTS','CPTEVENTS', 'DATETIMEEVENTS', 
		'INPUTEVENTS_CV', 'INPUTEVENTS_MV','PROCEDUREEVENTS_MV', 'MICROBIOLOGYEVENTS']
	MM_NAMES = ['ITEMID', 'ITEMID', 'CPT_CD', 'ITEMID', 'ITEMID', 'ITEMID', 'ITEMID']

	admission = pd.read_csv(os.path.join(DATA_PATH, 'ADMISSIONS.csv'))
	patients = pd.read_csv(os.path.join(DATA_PATH, 'PATIENTS.csv'))
	diagnose = pd.read_csv(os.path.join(DATA_PATH, 'DIAGNOSES_ICD.csv'))
	ccs = pd.read_csv(os.path.join(DATA_PATH, 'ccs_multi_dx_tool_2015.csv'))

	# records of each visit
	ad = select_admission(admission, BASIC_INFO)
	ad = cal_age(ad, patients)
	diag = diag_ccs(diagnose, ccs)
	icd_list = group_by(diag, 'ICD9_CODE', 'HADM_ID')
	ad = pd.merge(ad, icd_list, how='left', on='HADM_ID')
	ad = add_labels(ad, diag)
	ad = add_mm(ad, MULTIMODAL[:-1], MM_NAMES)
	ad = add_mm2(ad, MULTIMODAL[-1])
	ad.sort_values('HADM_ID',inplace=True)
	ad.to_pickle(os.path.join(SAVE_PATH, 'MIMIC.pkl'))

	# MIMIC_WHOLE
	ad_list = group_patients(ad, BASIC_INFO[2:], 'SUBJECT_ID')
	ad_list = group_mm_res(ad_list, ad, MULTIMODAL)
	ad_list = add_labels_res(ad_list, ad)
	ad_list = add_patients(ad_list, patients)
	ad_list = add_medical_history(ad_list, ad)
	ad_list.to_pickle(os.path.join(SAVE_PATH,'MIMIC_WHOLE.pkl'))

	# MIMIC_SEQ
	ad_list2 = ad_list[pd.notnull(ad_list['ICD9_CODE_DOCS'])]
	ad_list2.to_pickle(os.path.join(SAVE_PATH,'MIMIC_SEQ.pkl'))

	# Embedding
	ad.fillna('0',inplace=True)
	ad_list.fillna('0',inplace=True)
	ad_list2.fillna('0',inplace=True)

	[ad_list, ad_list2] = feild_embed([ad_list, ad_list2], BASIC_INFO[2:], embed_size=32)
	[ad_list, ad_list2] = feild_embed([ad_list, ad_list2], ['ICD9_CODE_DOCS'], win_size=50, embed_size=128)
	[_, ad_list, ad_list2] = feild_embed([ad, ad_list, ad_list2], MULTIMODAL, win_size=30, embed_size=128)

	ad_list.to_pickle(os.path.join(SAVE_PATH,'MIMIC_WHOLE_encode.pkl'))
	ad_list2.to_pickle(os.path.join(SAVE_PATH,'MIMIC_SEQ_encode.pkl'))
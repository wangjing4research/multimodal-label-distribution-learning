from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import backend as K

def EF_MLDL(seq_len, d_text, d_visual, d_audio, output_size, qk_hidden=32, v_hidden=128, lstm_hidden=128, ffn_hidden=128, dropout_rate=0.2):
    text_env = Input(shape=(seq_len,d_text))
    visual_env = Input(shape=(seq_len,d_visual))
    audio_env = Input(shape=(seq_len,d_audio))

    text = Input(shape=(seq_len,d_text))
    visual = Input(shape=(seq_len,d_visual))
    audio = Input(shape=(seq_len,d_audio))

    input_list = [text_env, visual_env, audio_env, text, visual, audio]

    env = Concatenate(axis=2)([text_env, visual_env, audio_env])
    env = Lambda(lambda x: K.expand_dims(x, axis=2))(env)
    query = Dropout(dropout_rate)(Dense(qk_hidden, activation='tanh')(env))
    key = Dropout(dropout_rate)(Dense(qk_hidden, activation='tanh')(env))
    qk = Lambda(lambda x:tf.matmul(x[0], x[1], transpose_a=True))([query, key])
    qk =  Lambda(lambda x: tf.reshape(x, (-1,seq_len,qk_hidden*qk_hidden)))(qk)
    
    dist = Dropout(dropout_rate)(Dense(3, activation='softmax')(qk))
    dist = Lambda(lambda x: K.expand_dims(x, axis=3))(dist)
    
    text = Dropout(dropout_rate)(Dense(v_hidden, activation='tanh')(text))
    text =  Lambda(lambda x: K.expand_dims(x, axis=2))(text)
    visual = Dropout(dropout_rate)(Dense(v_hidden, activation='tanh')(visual))
    visual =  Lambda(lambda x: K.expand_dims(x, axis=2))(visual)
    audio = Dropout(dropout_rate)(Dense(v_hidden, activation='tanh')(audio))
    audio =  Lambda(lambda x: K.expand_dims(x, axis=2))(audio)

    mm_values = Concatenate(axis=2)([text,visual,audio])
    d_values = Multiply()([dist, mm_values])
    d_out = Lambda(lambda x: tf.reshape(x, (-1,20,v_hidden*3)))(d_values)
    d_out = LayerNormalization(epsilon=1e-6)(d_out)

    mm_lstm = Dropout(dropout_rate)(Bidirectional(LSTM(lstm_hidden, return_sequences=True))(d_out))
    mm_lstm = TimeDistributed(Dense(lstm_hidden, activation="tanh"))(mm_lstm)    
    mm_max = Lambda(lambda x: K.max(x, axis=1))(mm_lstm)
    
    output = Dense(output_size, activation='softmax')(mm_max)
    model = Model(inputs=input_list, outputs=output)
    return model
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import tensorflow as tf

def LF_MLDL(basic_embed, mm_embed, icd_embed, basic_seq, mm_seq_len, icd_seq, output_size, 
    dropout_rate=0.2, lstm_hidden=128, qk_hidden=32, ffn_hidden=256):
    input_list = []
    basic_input = []
    for emb in basic_embed:
        emb_layer = Embedding(
                input_dim=emb.shape[0],
                output_dim=emb.shape[1],
                weights=[emb],
                trainable=False,
                mask_zero=True
            )
        input_seq = Input(shape=(basic_seq,))
        input_list.append(input_seq)
        embed_input = emb_layer(input_seq)
        basic_input.append(embed_input)
    
    basic_input = Concatenate(axis=2)(basic_input)
    basic_input = LayerNormalization(epsilon=1e-6)(basic_input)
    basic_lstm = Dropout(dropout_rate)(Bidirectional(LSTM(lstm_hidden, return_sequences=True))(basic_input))
    basic_lstm = TimeDistributed(Dense(lstm_hidden, activation="tanh"))(basic_lstm)
    basic_mean = Lambda(lambda x: K.mean(x, axis=1))(basic_lstm)
    
    mm_output = []
    for i in range(len(mm_embed)):
        emb = mm_embed[i]
        emb_layer = Embedding(
                input_dim=emb.shape[0],
                output_dim=emb.shape[1],
                weights=[emb],
                trainable=False,
                mask_zero=True
            )
        seq_len = mm_seq_len[i]
        input_seq = Input(shape=(seq_len,))
        input_list.append(input_seq)
        mm_input = emb_layer(input_seq)
        mm_lstm = Dropout(dropout_rate)(Bidirectional(LSTM(lstm_hidden, return_sequences=True))(mm_input))
        mm_lstm = TimeDistributed(Dense(lstm_hidden, activation="tanh"))(mm_lstm)
        mm_mean = Lambda(lambda x: K.mean(x, axis=1))(mm_lstm)
        mm_mean = Lambda(lambda x: K.expand_dims(x, axis=1))(mm_mean)
        mm_output.append(mm_mean)
    
    emb_layer = Embedding(
            input_dim=icd_embed.shape[0],
            output_dim=icd_embed.shape[1],
            weights=[icd_embed],
            trainable=False,
            mask_zero=True
        )
    input_seq = Input(shape=(icd_seq,))
    input_list.append(input_seq)
    icd_input = emb_layer(input_seq)   
    icd_lstm = Dropout(dropout_rate)(Bidirectional(LSTM(lstm_hidden, return_sequences=True))(icd_input))
    icd_lstm = TimeDistributed(Dense(lstm_hidden, activation="tanh"))(icd_lstm)
    icd_mean = Lambda(lambda x: K.mean(x, axis=1))(icd_lstm)
    icd_mean = Lambda(lambda x: K.expand_dims(x, axis=1))(icd_mean)

    mm_out = Concatenate(axis=1)(mm_output)
    mm_out = Concatenate(axis=1)([icd_mean, mm_out])
    mm_num = mm_out.shape[1]

    env = Lambda(lambda x: K.expand_dims(x, axis=1))(basic_mean)
    query = Dropout(dropout_rate)(Dense(qk_hidden, activation='tanh')(env))
    key = Dropout(dropout_rate)(Dense(qk_hidden, activation='tanh')(env))
    qk = Lambda(lambda x:tf.matmul(x[0], x[1], transpose_a=True))([query, key])
    qk =  Lambda(lambda x: tf.reshape(x, (-1,qk_hidden*qk_hidden)))(qk)
    
    dist = Dropout(dropout_rate)(Dense(mm_num, activation='softmax')(qk))
    dist = Lambda(lambda x: K.expand_dims(x, axis=2))(dist)
    d_out = Multiply()([dist, mm_out])
    
    d_out = Flatten()(d_out)    
    d_out = LayerNormalization(epsilon=1e-6)(d_out)
    
    ffn_out = Dense(ffn_hidden, activation='tanh')(d_out)
    output = Dense(output_size, activation='sigmoid')(ffn_out)
    model = Model(inputs=input_list, outputs=output)
    return model
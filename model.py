from keras.models import Model
from keras.layers import TimeDistributed,RepeatVector,Input, LSTM, Merge, Lambda, Masking,Reshape
from keras.layers.merge import Concatenate, Dot
import keras.backend as K
from keras import initializers,regularizers,constraints
from keras.engine.topology import Layer


class AttentionLayer(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        self.W = self.add_weight((input_shape[0][-1], input_shape[0][-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[0][-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        super(AttentionLayer, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def call(self,inputs, mask=None):
        x = inputs[0]     # Shape of x is (None,250,300)
        y = inputs[1]     # Shape of y is (None,300)
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)               # shape of uit is (None,250,300)
        y = K.expand_dims(y,axis=1)     # shape of y is (None,1,300)
        ait = K.sum(uit*y, axis=2)      # shape of ait is (None,250)
        a = K.exp(ait)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number  to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())   # shape of a is (None,250)
        a = K.expand_dims(a)     # shape of a is (None,250,1)
        weighted_input = x * a   # shape of weighted input is (None,250,300)
        return K.sum(weighted_input, axis=1)  # shape is (None, 300)
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

class tqa_model():
    def __init__(self,word_vec_size,max_q_length,max_doc_length,max_option_length,max_opt_count):
        self.max_q_length = max_q_length
        self.max_doc_length = max_doc_length
        self.max_option_length = max_option_length
        self.max_opt_count = max_opt_count
        # max_q_length=20
        # max_doc_length = 250
        # max_option_length = 50
        # max_opt_count = 7

    def get_non_masked_model(self):
        lstm_qo = LSTM(300)

        sum_dim1 = Lambda(lambda xin: K.sum(xin, axis = 1)/self.max_opt_count, output_shape=(self.max_opt_count,))

        q_input = Input(shape=(self.max_q_length,300),name='question_input')

        lstm_q = lstm_qo(q_input)

        lstm_q_rep=RepeatVector(self.max_doc_length)(lstm_q)
        doc_input = Input(shape=(self.max_doc_length,300),name='doc_input')


        doc_q = Concatenate(axis=-1)([doc_input,lstm_q_rep])
        lstm_doc = LSTM(300)(doc_q)

        option_input = Input(shape=(self.max_opt_count,self.max_option_length,300), name='option_input')

        lstm_td_opt = TimeDistributed(lstm_qo)(option_input)
        lstm_doc_rep = RepeatVector(self.max_opt_count)(lstm_doc)
        cossim = Dot(axes=2,normalize=True)([lstm_doc_rep,lstm_td_opt])
        sim1 = sum_dim1(cossim)
        main_model = Model(inputs=[q_input,doc_input,option_input],outputs=sim1)
        main_model.compile(loss='mse',optimizer='sgd')
        main_model.summary()
        return main_model


    def get_masked_model(self):
        lstm_qo = LSTM(300)

        sum_dim1 = Lambda(lambda xin: K.sum(xin, axis = 1)/self.max_opt_count, output_shape=(self.max_opt_count,))

        q_input = Input(shape=(self.max_q_length,300),name='question_input')
        q_mask = Masking(mask_value=0.)(q_input)
        lstm_q = lstm_qo(q_mask)

        lstm_q_rep=RepeatVector(self.max_doc_length)(lstm_q)
        doc_input = Input(shape=(self.max_doc_length,300),name='doc_input')
        doc_mask = Masking(mask_value=0.)(doc_input)

        doc_q = Concatenate(axis=-1)([doc_mask,lstm_q_rep])
        lstm_doc = LSTM(300)(doc_q)

        option_input = Input(shape=(self.max_opt_count,self.max_option_length,300), name='option_input')
        option_mask = TimeDistributed(Masking(mask_value=0.))(option_input)
        lstm_td_opt = TimeDistributed(lstm_qo)(option_mask)
        lstm_doc_rep = RepeatVector(self.max_opt_count)(lstm_doc)
        cossim = Dot(axes=2,normalize=True)([lstm_doc_rep,lstm_td_opt])
        sim1 = sum_dim1(cossim)
        main_model = Model(inputs=[q_input,doc_input,option_input],outputs=sim1)
        main_model.compile(loss='mse',optimizer='sgd')
        main_model.summary()
        return main_model

    def get_model_show_and_tell(self):
        lstm_qo = LSTM(300)

        sum_dim1 = Lambda(lambda xin: K.sum(xin, axis=1) / self.max_opt_count, output_shape=(self.max_opt_count,))

        q_input = Input(shape=(self.max_q_length, 300), name='question_input')
        lstm_q = lstm_qo(q_input)
        lstm_q = Reshape((1, 300))(lstm_q)
        doc_input = Input(shape=(self.max_doc_length, 300), name='doc_input')

        doc_q = Concatenate(axis=1)([lstm_q, doc_input])
        lstm_doc = LSTM(300)(doc_q)

        option_input = Input(shape=(self.max_opt_count, self.max_option_length, 300), name='option_input')
        lstm_td_opt = TimeDistributed(lstm_qo)(option_input)
        lstm_doc_rep = RepeatVector(self.max_opt_count)(lstm_doc)
        cossim = Dot(axes=2, normalize=True)([lstm_doc_rep, lstm_td_opt])
        sim1 = sum_dim1(cossim)
        main_model = Model(inputs=[q_input, doc_input, option_input], outputs=sim1)
        main_model.compile(loss='mse', optimizer='sgd')
        main_model.summary()


    def get_attention_model(self):
        lstm_qo = LSTM(300)
        sum_dim1 = Lambda(lambda xin: K.sum(xin, axis=1) / self.max_opt_count, output_shape=(self.max_opt_count,))

        q_input = Input(shape=(self.max_q_length, 300), name='question_input')
        lstm_q = lstm_qo(q_input)

        doc_input = Input(shape=(self.max_doc_length, 300), name='doc_input')
        lstm_doc = LSTM(300, return_sequences=True)(doc_input)

        lstm_doc = AttentionLayer()([lstm_doc, lstm_q])

        option_input = Input(shape=(self.max_opt_count, self.max_option_length, 300), name='option_input')
        lstm_td_opt = TimeDistributed(lstm_qo)(option_input)
        lstm_doc_rep = RepeatVector(self.max_opt_count)(lstm_doc)
        cossim = Dot(axes=2, normalize=True)([lstm_doc_rep, lstm_td_opt])
        sim1 = sum_dim1(cossim)
        main_model = Model(inputs=[q_input, doc_input, option_input], outputs=sim1)
        main_model.compile(loss='mse', optimizer='sgd')
        main_model.summary()



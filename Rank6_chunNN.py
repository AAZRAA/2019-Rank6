import os
import pickle
import re
import time
import numpy as np
import pandas as pd
from keras.activations import softmax
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import Word2Vec
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers
from keras import Input, Model
from keras.layers import *
from keras.layers import TimeDistributed, Embedding, Dot,Subtract,Multiply,Masking,subtract, multiply,Reshape,Conv1D,Convolution1D,MaxPool1D,GlobalMaxPooling1D, MaxPooling1D, Dense, Input, concatenate, SpatialDropout1D, PReLU,CuDNNGRU,CuDNNLSTM, BatchNormalization, Dropout, Lambda, Flatten
from keras.layers import Reshape, RepeatVector, Flatten, Lambda, Concatenate, Add, Softmax
import datetime
from tqdm import tqdm
from keras.utils import plot_model
from sklearn import metrics, preprocessing
from multiprocessing import cpu_count,Pool
import matplotlib.pyplot as plt
import gc

# 后面加载训练好的w2v模型时也需要有这个类的定义, 否则load会报找不到这个类的错误
class EpochSaver(CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" %
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)

        self.pre_loss = cum_loss
        self.since = time.time()
        

# 这是装饰函数
def timer(func):
    def wrapper(*args, **kw):
        t1=time.time()
        # 这是函数真正执行的地方
        argement=func(*args, **kw)
        t2=time.time()
        # 计算下时长
        cost_time = t2-t1
        print("{}功能:花费时间：{}秒".format(func.__name__,cost_time))
        return argement
    return wrapper
    
    
def pool_text_to_Sequence(tok,data,chunk_size,worker=-1):
    cpu_worker = cpu_count()
    print('cpu 核心有：{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker

    t1 = time.time()
    len_data = len(data)
    start = 0
    end = 0
    p = Pool(worker)
    res=[]
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        res.append(p.apply_async(tok.texts_to_sequences, (data[start:end],)))
        start = end
    p.close()
    p.join()
    t2 = time.time()
    print(t2 - t1)

    data = np.concatenate([i.get() for i in res], axis=0)
    return data


def pool_padding_Sequence(data,max_len,chunk_size,worker=-1):
    cpu_worker = cpu_count()
    print('cpu 核心有：{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker

    t1 = time.time()
    len_data = len(data)
    start = 0
    end = 0
    p = Pool(worker)
    res=[]
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        res.append(p.apply_async(pad_sequences,(data[start:end],max_len)))
        start = end
    p.close()
    p.join()
    t2 = time.time()
    print(t2 - t1)

    data = np.concatenate([i.get() for i in res], axis=0)
    return data


@timer
def load_data(train, test, model_vec_path, token_path, embedding_path, vector_size=100, q_len=12, t_len=22):
    all_content = pd.concat([train['query'], train['title'], test['query'], test['title']], axis=0)
    all_content = all_content.drop_duplicates()

    all_query = pd.concat([train['query'], test['query']], axis=0, ignore_index=True)
    all_title = pd.concat([train['title'], test['title']], axis=0, ignore_index=True)
    print('token')

    @timer
    def token_fun(token_path, all_content):
        if not os.path.exists(token_path):
            tok = Tokenizer(num_words=None)
            tok.fit_on_texts(all_content)
            with open(token_path, 'wb') as handle:
                pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)  #protocol使用最高版的协议
        else:
            with open(token_path, 'rb') as handle:
                tok = pickle.load(handle)
        return tok

    tok = token_fun(token_path, all_content)
    sequences_query = tok.texts_to_sequences(all_query)
    sequences_title = tok.texts_to_sequences(all_title)
    word_index = tok.word_index

    del all_query,all_title
    gc.collect()


    # embedding matrix
    @timer
    def embedding_matrix_fun(word_index, model_vec_path):
        print('emb matrix')
        embedding_matrix_size = len(word_index) + 1
        print('embedding矩阵大小 {}'.format(embedding_matrix_size))
        if not os.path.exists(embedding_path):
            model = Word2Vec.load(model_vec_path)
            embedding_matrix = np.zeros((embedding_matrix_size, vector_size))
            for word, index in tqdm(word_index.items()):
                if word in model:
                    embedding_matrix[index] = model[word]
            embedding_matrix.tofile(embedding_path)
        else:
            embedding_matrix = np.fromfile(embedding_path, dtype=np.float)
            embedding_matrix.shape = [embedding_matrix_size, vector_size]
        return embedding_matrix

    embedding_matrix = embedding_matrix_fun(word_index, model_vec_path)
    print('padding')
    query_all = pad_sequences(sequences_query, maxlen=q_len)
    title_all = pad_sequences(sequences_title, maxlen=t_len)
    len_train = len(train)
    train_query = query_all[:len_train]
    test_query = query_all[len_train:]
    train_title = title_all[:len_train]
    test_title = title_all[len_train:]
    return train_query, train_title, test_query, test_title, embedding_matrix

def pandas_reduce_mem_usage(df):
    start_mem=df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    starttime = datetime.datetime.now()
    for col in df.columns:
        col_type=df[col].dtype   #每一列的类型
        if col_type !=object:    #不是object类型
            c_min=df[col].min()
            c_max=df[col].max()
            # print('{} column dtype is {} and begin convert to others'.format(col,col_type))
            if str(col_type)[:3]=='int':
                #是有符号整数
                if c_min<0:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min >= np.iinfo(np.uint8).min and c_max<=np.iinfo(np.uint8).max:
                        df[col]=df[col].astype(np.uint8)
                    elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
            #浮点数
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # print('\t\tcolumn dtype is {}'.format(df[col].dtype))

        #是object类型，比如str
        else:
            # print('\t\tcolumns dtype is object and will convert to category')
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    endtime = datetime.datetime.now()
    print('consume times: {:.4f}'.format((endtime - starttime).seconds))
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


if __name__=='__main__':
    from numpy.random import seed
    seed(1)
    train1_path='/home/kesci/work/Rematch/data/train_back_2d5qw.csv.gz'
    test_path='/home/kesci/input/bytedance/test_final_part1.csv'

    vec_path='/home/kesci/work/Rematch/data/model/fasttext/FastText_100m.model'
    token_path='/home/kesci/work/Rematch/data/shendu_chun/em_save/token_7271.pickle'
    embedding_path='/home/kesci/work/Rematch/data/shendu_chun/em_save/embedding_7271.bin'
    save_path='/home/kesci/work/Rematch/data/shendu_chun/best_model/chun_2qw_esim_7251.h5.h5'
    save_best_path='/home/kesci/work/Rematch/data/shendu_chun/best_model/chun_2qw_esim_best7251.h5.h5'
        
    q_len = 12
    t_len = 22

    size=-5000000
    train = pd.read_csv(train1_path, header=None,
                       names=['query_id', 'query', 'query_title_id', 'title', 'label'])
    train_preds=train[['query_id','label']].iloc[:size,:]
    validate_preds=train[['query_id','label']].iloc[size:,:]
    train_y=train[['label']].iloc[:size,:].values
    validate_y=train[['label']].iloc[size:,:].values
    
    test= pd.read_csv(test_path, header=None,
                       names=['query_id', 'query', 'query_title_id', 'title'])
    test['title']=test['title'].apply(lambda x: x[:-1])
    test_preds=test[['query_id','query_title_id']]


    print(train.shape)
    train_query, train_title,test_query,test_title ,embedding_matrix=load_data(
                    train,test,vec_path,token_path,embedding_path,vector_size=100,q_len=q_len,t_len=t_len)
    validate_query=train_query[size:]
    train_query=train_query[:size]
    validate_title=train_title[size:]
    train_title=train_title[:size]
  
    del train,test
    gc.collect()


    def create_pretrained_embedding(pretrained_weights, trainable=False, **kwargs):
    """Create embedding layer from a pretrained weights array"""
    pretrained_weights = pretrained_weights
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=trainable, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def substract(input_1, input_2):
    """Substract element-wise"""
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    """Get multiplication and subtraction then concatenate results"""
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    """Apply layers to input then concatenate result"""
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    """Apply a list of layers in TimeDistributed mode"""
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

class deep1_model(object):
    def __init__(self, maxlen1,maxlen2, max_features, embedding_dims,
                 class_num=1,em=None,
                 last_activation='sigmoid'):
        self.maxlen_query = maxlen1   #文本最大长度
        self.maxlen_title = maxlen2     #title最大长度
        self.max_features = max_features              #词库最大的个数
        self.embedding_dims = embedding_dims        #词向量的维度
        self.class_num = class_num                     #输出维度
        self.last_activation = last_activation         #最后的激活函数
        self.embedding_ma=em
        self.model=self.get_model()

    def text_encoder(self, input_shape):
        seed = 1
        text_input = Input(shape=(input_shape,))
        embedding_layer = Embedding(self.max_features, self.embedding_dims, input_length=input_shape,
                                    weights=[self.embedding_ma],
                                    trainable=False)
        x = embedding_layer(text_input)

        return Model(text_input, [x])
        

    def get_model(self):
        seed = 1
        input1 = Input(shape=(self.maxlen_query,))
        input2 = Input(shape=(self.maxlen_title,))

        text_encode1 = self.text_encoder(self.maxlen_query)
        text_encode2 = self.text_encoder(self.maxlen_title)

        # interaction
        query_embedding = text_encode1(input1)
        title_embedding = text_encode2(input2)
        
        q_embed = query_embedding
        t_embed = title_embedding
        
        ###########################################model1
        # 迭代单词向量的1D卷积
        conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
        conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
        conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
        conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
        conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')
        
        # Run through CONV + GAP layers
        conv1a = conv1(query_embedding)
        glob1a = GlobalAveragePooling1D()(conv1a)
        conv1b = conv1(title_embedding)
        glob1b = GlobalAveragePooling1D()(conv1b)
        
        conv2a = conv2(query_embedding)
        glob2a = GlobalAveragePooling1D()(conv2a)
        conv2b = conv2(title_embedding)
        glob2b = GlobalAveragePooling1D()(conv2b)
        
        conv3a = conv3(query_embedding)
        glob3a = GlobalAveragePooling1D()(conv3a)
        conv3b = conv3(title_embedding)
        glob3b = GlobalAveragePooling1D()(conv3b)
        
        conv4a = conv4(query_embedding)
        glob4a = GlobalAveragePooling1D()(conv4a)
        conv4b = conv4(title_embedding)
        glob4b = GlobalAveragePooling1D()(conv4b)
        
        conv5a = conv5(query_embedding)
        glob5a = GlobalAveragePooling1D()(conv5a)
        conv5b = conv5(title_embedding)
        glob5b = GlobalAveragePooling1D()(conv5b)
        
        conv6a = conv6(query_embedding)
        glob6a = GlobalAveragePooling1D()(conv6a)
        conv6b = conv6(title_embedding)
        glob6b = GlobalAveragePooling1D()(conv6b)
        
        mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
        mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])
        
        merge1 = concatenate([glob1a, glob1b])
        merge2 = concatenate([glob2a, glob2b])
        merge3 = concatenate([glob3a, glob3b])
        merge4 = concatenate([glob4a, glob4b])
        merge5 = concatenate([glob5a, glob5b])
        merge6 = concatenate([glob6a, glob6b])
        mergec = concatenate([merge1, merge2, merge3, merge4, merge5, merge6])
        
        # 采用两个句子之间明确的绝对差异
        # 采用乘法不同的条目来得到不同的均衡度量
        diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
        mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
        

        # Merge the Magic and distance features with the difference layer
        merge = concatenate([diff, mul, mergec])
        # merge = concatenate([diff, mul, magic_dense, distance_dense])
        
        # The MLP that determines the outcome
        # x = Dropout(0.2)(merge)
        # x = BatchNormalization()(x)
        # x = Dense(300, activation='relu')(x)
        
        # x = Dropout(0.2)(x)
        # x = BatchNormalization()(x)
        
        projection_dim=128  #300
        projection_hidden=0
        projection_dropout=0.2
        compare_dim=256   #500
        compare_dropout=0.2
        dense_dim=300
        dense_dropout=0.2
        lr=1e-3
        activation='relu'
        query_len=12
        title_len=22
        
        ##########################################model2
        projection_layers = []
        if projection_hidden > 0:
            projection_layers.extend([
                Dense(projection_hidden, activation=activation),
                Dropout(rate=projection_dropout),
            ])
        projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
        q1_encoded = time_distributed(q_embed, projection_layers)
        q2_encoded = time_distributed(t_embed, projection_layers)
    
        # Attention
        q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
        # Compare
        q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
        q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
        compare_layers = [
            Dense(compare_dim, activation=activation),
            Dropout(compare_dropout),
            Dense(compare_dim, activation=activation),
            Dropout(compare_dropout),
        ]
        q1_compare = time_distributed(q1_combined, compare_layers)
        q2_compare = time_distributed(q2_combined, compare_layers)
    
        # Aggregate
        q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
        # Classifier
        merged = Concatenate()([q1_rep, q2_rep])
        
        
        
        ############################################model3
        bn = BatchNormalization(axis=2)
        q1_embed = bn(q_embed)
        q2_embed = bn(t_embed)
        encoded = Bidirectional(CuDNNLSTM(64, return_sequences=True))
        q1_encodedd = encoded(q1_embed)
        q2_encodedd = encoded(q2_embed)
        
        # Attention
        q1_alignedd, q2_alignedd = soft_attention_alignment(q1_encodedd, q2_encodedd)
        
        # Compose
        q1_combinedd = Concatenate()([q1_encodedd, q2_alignedd, submult(q1_encodedd, q2_alignedd)])
        q2_combinedd = Concatenate()([q2_encodedd, q1_alignedd, submult(q2_encodedd, q1_alignedd)]) 
           
        composed = Bidirectional(CuDNNLSTM(64, return_sequences=True))
        q1_compared = composed(q1_combinedd)
        q2_compared = composed(q2_combinedd)
        
        # Aggregate
        q1_repp = apply_multiple(q1_compared, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_repp = apply_multiple(q2_compared, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        
        # Classifier
        mergee = Concatenate()([q1_repp, q2_repp])
        
        #####################################################################
        x = concatenate([merge, merged, mergee])
        dense = BatchNormalization()(x)
        dense = Dense(dense_dim, activation=activation)(dense)
        dense = Dropout(dense_dropout)(dense)
        dense = BatchNormalization()(dense)
        dense = Dense(dense_dim, activation=activation)(dense)
        dense = Dropout(dense_dropout)(dense)
        # xx = dense
        
        
        # x = concatenate([x, xx])
        output = Dense(1, activation='sigmoid', name='output')(dense)

        model = Model(inputs=[input1, input2], outputs=output)
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()
        
        return model



    def train(self, train_query,train_title,train_y,batch_size=64, n_epochs=5,
              save_path='textcnn.h5',file_path='weight.best.h5'):

        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
        checkpoint=ModelCheckpoint(file_path,monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
        callback_list=[checkpoint,earlyStopping]
        his=self.model.fit([train_query,train_title], train_y, batch_size=batch_size,validation_split=0.2, epochs=n_epochs,
                       verbose=1,callbacks=callback_list)
        self.model.save(save_path)

        plt.plot(his.history['val_loss'])
        plt.plot(his.history['loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['validate', 'train'], loc='upper left')
        plt.show()


    def predict(self, test_query,test_title ,model_path='textcnn.h5'):
        model = self.model
        model.load_weights(model_path)
        preds = model.predict([test_query,test_title],verbose=1)
        return preds


# 初始化模型
text_cnn = deep1_model(q_len, t_len, len(embedding_matrix), 100, class_num=1, em=embedding_matrix)
text_cnn.train(train_query, train_title, train_y, batch_size=512, n_epochs=10,
                   save_path=save_path, file_path=save_best_path)

from tqdm import tqdm
@timer
def get_new_auc(test):

    query=test[['query_id']].drop_duplicates()
    print('第一步')
    temp1=test[['query_id','label']]
    temp1=temp1.groupby('query_id').agg(lambda x:list(x)).reset_index().rename(columns={'label':'label_list'})
    query=pd.merge(query,temp1,on=['query_id'],how='left')
    query=query[query['label_list'].apply(lambda x: len(set(x))!=1)]
    print('第二步')
    temp2=test[['query_id','preb']]
    temp2 = temp2.groupby('query_id').agg(lambda x: list(x)).reset_index().rename(columns={'preb': 'preb_list'})
    query = pd.merge(query, temp2, on=['query_id'], how='left')
    print('第三步')
    query['q_auc']=list(map(lambda x,y: metrics.roc_auc_score(x,y),tqdm(query['label_list']),query['preb_list']))
    return  query['q_auc'].mean()

#加载模型
text_cnn=deep1_model(q_len,t_len,len(embedding_matrix),100,class_num=1,em=embedding_matrix)
text_cnn.model.load_weights(save_best_path)

#验证集
validate_preds['preb']=text_cnn.predict(validate_query,validate_title,model_path=save_best_path)
print(validate_preds.head(10))
print(get_new_auc(validate_preds))
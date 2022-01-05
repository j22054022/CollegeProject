# -*- coding：utf-8 -*-
# -*- author：zzZ_CMing  CSDN address:https://blog.csdn.net/zzZ_CMing
# -*- 2018/07/31；14:23
# -*- python3.5
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import word_token
import jieba
import random
import os
import json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

size = 8               # LSTM神經元size
GO_ID = 1              # 輸出序列起始標記
EOS_ID = 2             # 結尾標記
PAD_ID = 0             # 空值填充0
min_freq = 1           # 樣本頻率超過這個值纔會存入詞表
epochs = 2000          # 訓練次數
batch_num = 1000       # 參與訓練的問答對個數
input_seq_len = 25         # 輸入序列長度
output_seq_len = 50        # 輸出序列長度
init_learning_rate = 0.5     # 初始學習率

testlist=[]

wordToken = word_token.WordToken()

# 放在全局的位置，爲了動態算出 num_encoder_symbols 和 num_decoder_symbols
max_token_id = wordToken.load_file_list(['./samples/question', './samples/answer'], min_freq)
num_encoder_symbols = max_token_id + 5
num_decoder_symbols = max_token_id + 5
 
def get_id_list_from(sentence):
    """
    得到分詞後的ID
    """
    sentence_id_list = []
    seg_list = jieba.cut(sentence)
    for str in seg_list:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(wordToken.word2id(str))
    return sentence_id_list


def get_train_set():
    """
    得到訓練問答集
    """
    global num_encoder_symbols, num_decoder_symbols
    train_set = []
    with open('./samples/question', 'r', encoding='utf-8') as question_file:
        with open('./samples/answer', 'r', encoding='utf-8') as answer_file:
            while True:
                question = question_file.readline()
                answer = answer_file.readline()
                if question and answer:
                    # strip()方法用於移除字符串頭尾的字符
                    question = question.strip()
                    answer = answer.strip()

                    # 得到分詞ID
                    question_id_list = get_id_list_from(question)
                    answer_id_list = get_id_list_from(answer)
                    if len(question_id_list) > 0 and len(answer_id_list) > 0:
                        answer_id_list.append(EOS_ID)
                        train_set.append([question_id_list, answer_id_list])
                else:
                    break
    return train_set


def get_samples(train_set, batch_num):
    """
    構造樣本數據:傳入的train_set是處理好的問答集
    batch_num:讓train_set訓練集裏多少問答對參與訓練
    """
    raw_encoder_input = []
    raw_decoder_input = []
    if batch_num >= len(train_set):
        batch_train_set = train_set
    else:
        random_start = random.randint(0, len(train_set)-batch_num)
        batch_train_set = train_set[random_start:random_start+batch_num]

    # 添加起始標記、結束填充
    for sample in batch_train_set:
        raw_encoder_input.append([PAD_ID] * (input_seq_len - len(sample[0])) + sample[0])
        raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []

    for length_idx in range(input_seq_len):
        encoder_inputs.append(np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))
    for length_idx in range(output_seq_len):
        decoder_inputs.append(np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
        target_weights.append(np.array([
            0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == PAD_ID else 1.0 for decoder_input in raw_decoder_input
        ], dtype=np.float32))
    return encoder_inputs, decoder_inputs, target_weights


def seq_to_encoder(input_seq):
    """
    從輸入空格分隔的數字id串，轉成預測用的encoder、decoder、target_weight等
    """
    input_seq_array = [int(v) for v in input_seq.split()]
    encoder_input = [PAD_ID] * (input_seq_len - len(input_seq_array)) + input_seq_array
    decoder_input = [GO_ID] + [PAD_ID] * (output_seq_len - 1)
    encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
    decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
    target_weights = [np.array([1.0], dtype=np.float32)] * output_seq_len
    return encoder_inputs, decoder_inputs, target_weights


def get_model(feed_previous=False):
    """
    構造模型
    """
    learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    # decoder_inputs左移一個時序作爲targets
    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]

    cell = tf.contrib.rnn.BasicLSTMCell(size)

    # 這裏輸出的狀態我們不需要
    outputs, _ = seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs[:output_seq_len],
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=feed_previous,
                        dtype=tf.float32)

    # 計算加權交叉熵損失
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    # 梯度下降優化器
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # 優化目標：讓loss最小化
    update = opt.apply_gradients(opt.compute_gradients(loss))
    # 模型持久化
    saver = tf.train.Saver(tf.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate


def train():
    """
    訓練過程
    """
    train_set = get_train_set()
    with tf.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model()
        sess.run(tf.global_variables_initializer())

        # 訓練很多次迭代，每隔100次打印一次loss，可以看情況直接ctrl+c停止
        previous_losses = []
        for step in range(epochs):
            sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples(train_set, batch_num)
            input_feed = {}
            for l in range(input_seq_len):
                input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
            for l in range(output_seq_len):
                input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                input_feed[target_weights[l].name] = sample_target_weights[l]
            input_feed[decoder_inputs[output_seq_len].name] = np.zeros([len(sample_decoder_inputs[0])], dtype=np.int32)
            [loss_ret, _] = sess.run([loss, update], input_feed)
            if step % 100 == 0:
                print('step=', step, 'loss=', loss_ret, 'learning_rate=', learning_rate.eval())
                #print('333', previous_losses[-5:])

                if len(previous_losses) > 5 and loss_ret > max(previous_losses[-5:]):
                    sess.run(learning_rate_decay_op)
                previous_losses.append(loss_ret)

                # 模型參數保存
                saver.save(sess, './model/'+ str(epochs)+ '/demo_')
                #saver.save(sess, './model/' + str(epochs) + '/demo_' + step)

"""
def predict():
    
    with tf.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model(feed_previous=True)
        saver.restore(sess, './model/'+str(epochs)+'/demo_')
        sys.stdout.write("you ask>> ")
        sys.stdout.flush()
        input_seq = sys.stdin.readline()
        while input_seq:
            input_seq = input_seq.strip()
            input_id_list = get_id_list_from(input_seq)
            if (len(input_id_list)):
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = seq_to_encoder(' '.join([str(v) for v in input_id_list]))

                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

                # 預測輸出
                outputs_seq = sess.run(outputs, input_feed)
                # 因爲輸出數據每一個是num_decoder_symbols維的，因此找到數值最大的那個就是預測的id，就是這裏的argmax函數的功能
                outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
                # 如果是結尾符，那麼後面的語句就不輸出了
                if EOS_ID in outputs_seq:
                    outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
                outputs_seq = [wordToken.id2word(v) for v in outputs_seq]
                print("chatbot>>", " ".join(outputs_seq))
            else:
                print("WARN：詞彙不在服務區")

            sys.stdout.write("you ask>>")
            sys.stdout.flush()
            input_seq = sys.stdin.readline()
"""


def predict():

    with tf.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model(feed_previous=True)
        saver.restore(sess, './model/'+str(epochs)+'/demo_')
        init_time=os.stat('datajson.json').st_mtime
        buff_time=init_time
        while True:
            time = os.stat('datajson.json').st_mtime
            if time!=init_time and time!=buff_time:
                buff_time=time
                with open('datajson.json', 'r') as reader:
                    input_seq = json.loads(reader.read())
                    input_seq = input_seq['data']
                input_seq = input_seq.strip()
                input_id_list = get_id_list_from(input_seq)
                if (len(input_id_list)):
                    sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = seq_to_encoder(' '.join([str(v) for v in input_id_list]))

                    input_feed = {}
                    for l in range(input_seq_len):
                        input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                    for l in range(output_seq_len):
                        input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                        input_feed[target_weights[l].name] = sample_target_weights[l]
                    input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

                    # 預測輸出
                    outputs_seq = sess.run(outputs, input_feed)
                    # 因爲輸出數據每一個是num_decoder_symbols維的，因此找到數值最大的那個就是預測的id，就是這裏的argmax函數的功能
                    outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
                    # 如果是結尾符，那麼後面的語句就不輸出了
                    if EOS_ID in outputs_seq:
                        outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
                    outputs_seq = [wordToken.id2word(v) for v in outputs_seq]
                    print("chatbot>>", " ".join(outputs_seq))
                    writeanswer(" ".join(outputs_seq))
                else:
                    writeanswer("不好意思我聽不懂")
                    print("不好意思我聽不懂")
            else:
                continue

def writeanswer(answerlist):
    alternativelist = {'answer': answerlist}
    with open('answerjson.json') as f:
        bufferlist = json.load(f)
    print(bufferlist)
    bufferlist = alternativelist
    print(bufferlist)
    with open('answerjson.json', 'w') as f:
        json.dump(bufferlist, f)

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    else:
        predict()

    #else:
        #print(predict2("今天吃了嗎"))
        #print(predict("你叫"))

# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_
import nltk

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    # encoder_length
    def __init__(self):
        self.encoder_inputs = [] #PAD之后的序列，嵌套列表，每个元素是一个句子中单词的Id
        self.encoder_inputs_length = [] #PAD之后序列的长度，动态编解码时序列的长度，一维列表，每个元素对应每个句子的长度
        self.decoder_targets = []
        self.decoder_targets_length = []


def load_data(filepath):
    '''
    加载数据
    :param filepath: 数据路径
    :return: data
    '''
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            data.append(line)
    return data


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def createBatch(samples):
    '''
    padding:填充
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    # 获取每个样本的长度，并保存在source_length和target_length中
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) + 1 for sample in samples]


    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        # 反序是经验所得，可能是因为序列开始信息
        # 将source进行反序并PAD值，并计算本batch的最大长度和应该补充的0的长度
        source = list(reversed(sample[0]))
        pad = [padToken] * (max_source_length - len(source)) #这里是保证batch长度，对于不相等的进行补0操作
        batch.encoder_inputs.append(pad + source)

        # 将target进行PAD，并添加END符号
        target = sample[1]
        pad = [padToken] * (max_target_length - len(target) - 1)
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)
        # batch.decoder_targets.append(target + pad)
        #encoder和decoder是上下一句的关系，同时encoder反序，decoder正序

    return batch


def getBatches(processed_data, batch_size):
    batches = []
    data_len = len(processed_data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield processed_data[i:min(i + batch_size, data_len)] #generator 像是中断函数

    # tensorflow中有batch函数，
    # tf.train.batch()
    for samples in genNextSamples():
        # samples有batchs_size行，每行是一个句子
        batch = createBatch(samples) #samples是一个batch size的数据，生成的batch可以直接传入feed_dict
        batches.append(batch)
    return batches


def process_all_data(data):
    '''
    得到输入和输出的字符映射表
    :param data: 原始数据，内容为汉字
    :return: processed_data: 映射后的数据，内容为数字
             word_to_id: 字典，数字到数字的转换
             id_to_word: 字典，数字到汉字的转换
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([character for line in data for subline in line for character in subline]))
    id_to_word = {idx: word for idx, word in enumerate(special_words + set_words)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将每一行转换成字符id的list
    processed_data = [[[word_to_id.get(word, word_to_id['<UNK>'])
                   for word in subline] for subline in line] for line in data]

    return processed_data, word_to_id, id_to_word


def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    # 分词
    # tokens = nltk.word_tokenize(sentence)
    tokens = ''.join(sentence.split())

    # 将每个单词转化为id，这里输入只有一句直接转换就好了。
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    # 调用createBatch构造batch
    batch = createBatch([[wordIds, []]])
    return batch


if __name__ == '__main__':
    filepath = 'data/data.txt'
    batch_size = 70
    data = load_data(filepath)
    processed_data, word_to_id, id_to_word = process_all_data(data)  # 根据词典映射
    batches = getBatches(processed_data, batch_size)

    temp = 0
    for nexBatch in batches:
        if temp == 0:
            print(len(nexBatch.encoder_inputs))
            print(len(nexBatch.encoder_inputs_length))
            print(nexBatch.decoder_targets)
            print(nexBatch.decoder_targets_length)
        temp += 1











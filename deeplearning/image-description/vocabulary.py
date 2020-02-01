import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):

    def __init__(self,
        vocab_threshold,
        vocab_file='./vocab.pkl',
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file='./annotations/captions_train2014.json',
        vocab_from_file=False):
        """初始化词汇表.
        Args:
          vocab_threshold: 最小单词数阈值.在图像标注中单词出现次数少于vocab_threshold的被视为unknown
          vocab_file: 文件存储路径.
          start_word: 单词开始的标志符号.
          end_word: 单词结尾的标志符号.
          unk_word: 未知单词的标志符号.
          annotations_file: 训练数据文件路径.
          vocab_from_file: 如果设置为False, 重头创建一个词库数据库文件，或者覆盖原有的文件
                           如果设置为True, 直接加载存在的词库数据库文件
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
		#从词汇库文件中加载或者从头开始生成词汇库数据文件
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
        
    def build_vocab(self):
        #构造词库数据库文件
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
		#初始化词汇表
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        #添加单词到词汇表中
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
		#将标注中出现单词次数超过阈值的加入到词汇数据文件中
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
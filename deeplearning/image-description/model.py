import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1] #ResNet模型的除最后一层之外的所有层
        self.resnet = nn.Sequential(*modules)#构造一个无ResNet模型最后一层的新架构
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images) #features output shape: [batch,2048,1,1]
        features = features.view(features.size(0), -1)
        features = self.embed(features)#把图像的维度设置和word一样的维度，用做解码器的传入数据
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions, hc):
	    #训练用，测试没有标注
        if captions is not None:
            seq_len = captions.shape[1]
            captions_emd = self.word_embeddings(captions)
			#图片是序列的第一个输入，最后一个输入是最后一个序列的倒数第二个，倒数第一个是结束标识
            captions_emd[:,1:seq_len] = captions_emd[:,0:seq_len-1]
            captions_emd[:,0] = features
        #测试用			
        else:
            if features.shape[-1] == self.embed_size:
                captions_emd = features
            else:
                captions_emd = self.word_embeddings(features)
        
        output, (h, c) = self.lstm(captions_emd, hc)  
        output = self.dropout(output)
        output = self.fc(output)
        return output, (h, c)
    
    def sample(self, inputs, states=None, max_len=20):
	    #根据输入的图片，返回相应的文字说明
        if states is None:
            h = self.init_hidden(1)
        sentence = []
		#每遍历一次，给出一个单词，最大标注长度为max_len，或者遇到<end>标识符就结束
        for index in range(max_len):
            h = tuple([each.data for each in h])
            output, h = self.forward(inputs, None, h)
            output = output.view(-1)
            p = F.softmax(output).data
			#找出概率最高的单词，将该单词作为预测的结果
            word_p, word_id = p.topk(1)
            inputs = word_id.view(1,-1)
            sentence.append(word_id.item())
            
            if word_id == 1: # <end> over，如果遇到结尾标识，则标识结束
                break
        
        return sentence
    
    def init_hidden(self, n_seqs=1):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, n_seqs, self.hidden_size).zero_(),
                weight.new(self.num_layers, n_seqs, self.hidden_size).zero_())
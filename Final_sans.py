import torch, pdb
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        vgg_model = models.vgg19(pretrained=True)
        self.in_features = 512  # input size of feature vector
        model = nn.Sequential(
            *list(vgg_model.features.children())[:-1])    # remove last fc layer

        self.model = model                              # loaded model without last fc layer
        self.fc = nn.Linear(512, embed_size)    # feature vector of image
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        #print(img_feature.shape)
#        pdb.set_trace()
        img_feature = img_feature.view(-1, self.in_features)
        #print(img_feature.shape)
        #pdb.set_trace()
        img_feature = self.fc(img_feature).view(-1,196,1024)                   # [batch_size, embed_size]
        #print(img_feature.shape)
        #pdb.set_trace()
        img_feature = self.dropout(self.tanh(img_feature))

        return img_feature

class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout, use_gpu):
        super(QuestionEmbedding, self).__init__() # Must call super __init__()
        self.use_gpu = use_gpu
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.lookuptable = nn.Linear(vocab_size, emb_size, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                num_layers=num_layers, bias=True,
                batch_first=True, dropout=dropout,bidirectional=True)

        return

    def forward(self, ques_vec, ques_len):            # forward(self, ques_vec, ques_len) | ques_vec: [batch_size, 26]
        #pdb.set_trace()
        B, W = ques_vec.size()

        # Add 1 to vocab_size, since word idx from 0 to vocab_size inclusive
        one_hot_vec = torch.zeros(B, W, self.vocab_size+1).scatter_(2,
                        ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)

        # To remove additional column in one_hot, use slicing
        one_hot_vec = Variable(one_hot_vec[:,:,1:], requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            one_hot_vec = one_hot_vec.cuda()

        x = self.lookuptable(one_hot_vec)

        # emb_vec: [batch_size or B, 26 or W, emb_size]
        emb_vec = self.dropout(self.tanh(x))

        # h: [batch_size or B, 26 or W, hidden_size]
        self.LSTM.flatten_parameters()
        h_emb, _ = self.LSTM(emb_vec)
        #pdb.set_trace()
        a = torch.LongTensor(ques_len.cpu().numpy()-1)
        mask = torch.zeros(B, W).scatter_(1, a.view(-1, 1), 1)
        mask = Variable(mask.view(B, W, 1), requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            mask = mask.cuda()

        h_emb = h_emb.transpose(1,2)
        # print(h.size(), mask.size())

        # output: [B, hidden_size]
        return torch.bmm(h_emb, mask).view(B, -1)

class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature

class Attention(nn.Module): # Extend PyTorch's Module class
    def __init__(self, input_size, att_size, img_seq_size, output_size, drop_ratio):
        super(Attention, self).__init__() # Must call super __init__()
        self.input_size = input_size
        self.att_size = att_size
        self.img_seq_size = img_seq_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio

        self.tan = nn.Tanh()
        self.dp = nn.Dropout(drop_ratio)
        self.sf = nn.Softmax(dim=2)

        self.fc1_1a = nn.Linear(input_size, 768, bias=True)
        self.bn768 = nn.BatchNorm1d(768)
        self.fc1_1b = nn.Linear(768, 640, bias=True)
        self.bn640 = nn.BatchNorm1d(640)
        self.fc1_2a = nn.Linear(input_size, 768, bias=False)
        self.fc1_2b = nn.Linear(768, 640, bias=False)
        self.fc1_3a = nn.Linear(640, 1, bias=True)
        #self.fc1_3b = nn.Linear(att_size, 1, bias=True)

        self.fc2_1a = nn.Linear(input_size, 768, bias=True)
        self.fc2_1b = nn.Linear(768, 640, bias=True)
        self.fc2_2a = nn.Linear(input_size, 768, bias=False)
        self.fc2_2b = nn.Linear(768, 640, bias=False)
        self.fc2_3a = nn.Linear(640, 1, bias=True)
        #self.fc2_3b = nn.Linear(att_size, 1, bias=True)

        #self.fc2_1a = nn.Linear(input_size, att_size, bias=True)
        #self.fc2_2 = nn.Linear(input_size, att_size, bias=False)
        #self.fc23 = nn.Linear(att_size, 1, bias=True)

        self.fc = nn.Linear(input_size, output_size, bias=True)
        #self.fc2 = nn.Linear(output_size, output_size, bias=True) 
        # d = input_size | m = img_seq_size | k = att_size
    def forward(self, ques_feat, img_feat):  # ques_feat -- [batch, d] | img_feat -- [batch_size, m, d]
        #  print(img_feat.size(), ques_feat.size())
        B = ques_feat.size(0)

        # Stack 1
        #pdb.set_trace()
        ques_emb_1 = self.fc1_1a(ques_feat)
        ques_emb_1 = self.fc1_1b(ques_emb_1) # [batch_size, att_size]
        img_emb_1 = self.fc1_2a(img_feat)
        img_emb_1 = self.fc1_2b(img_emb_1)
        #img_emb_1 = self.fc1_3a(img_emb_1)
        #img_emb_1 = self.fc1_3b(img_emb_1)
        #print(img_emb_1.shape,ques_emb_1.shape)
        #t1 = self.fc1_3b(ques_emb_1)
        #print(t1.shape,img_emb_1.shape)
        #pdb.set_trace()
        h1 = self.tan(ques_emb_1.unsqueeze(1) + img_emb_1)#.view(B,1,640)  + img_emb_1)
        h1_emb = self.fc1_3a(h1)
        #pdb.set_trace() 
        sf1 = self.sf(h1_emb) 
        #pdb.set_trace()
        p1 = sf1.view(-1, self.img_seq_size).unsqueeze(1)#.view(B, 1, self.img_seq_size)

        # Weighted sum
        img_att1 = p1.matmul(img_feat)
        u1 = ques_feat + img_att1.view(-1, self.input_size) #(Question embedding aware with image features)

        # Stack 2
        ques_emb_2 = self.fc2_1a(u1)
        ques_emb_2 = self.fc2_1b(ques_emb_2)  # [batch_size, att_size]
        #ques_emb_2 = self.tan(self.dp(self.fc2_1b(ques_emb_2)))
        img_emb_2 = self.fc2_2a(img_feat)
        img_emb_2 = self.fc2_2b(img_emb_2)

        h2 = self.tan(ques_emb_2.unsqueeze(1) + img_emb_2)#.view(B,1,640) + img_emb_2)
        h2_emb = self.fc2_3a(h2)

        p2 = self.sf(h2_emb).view(-1, self.img_seq_size).unsqueeze(1)#view(B, 1, self.img_seq_size)
        #pdb.set_trace()
        # Weighted sum
        img_att2 = p2.matmul(img_feat)
        u2 = u1 + img_att2.view(-1, self.input_size)

        # score
        score = self.fc(u2)
        #score = self.fc2(score1)
        return score,p1,p2


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size,att_size,img_seq_size,use_gpu):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        #self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.qst_encoder = QuestionEmbedding(vocab_size = qst_vocab_size, emb_size = embed_size, hidden_size=hidden_size,num_layers= num_layers,dropout=0.5,use_gpu=use_gpu)
        self.att = Attention(embed_size,att_size,img_seq_size,ans_vocab_size, 0.5)
        #self.tanh = nn.Tanh()
        #self.dropout = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        #self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
    def forward(self, img, qst, qlen):

        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst,qlen)                     # [batch_size, embed_size]
        #print(img_feature.shape,qst_feature.shape)
        #pdb.set_trace()
        output,p1,p2 = self.att(qst_feature,img_feature)              # [batch_size, ans_vocab_size=1000] 
        #combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        #combined_feature = self.tanh(combined_feature)
        #combined_feature = self.dropout(combined_feature)
        #combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        #combined_feature = self.tanh(combined_feature)
        #combined_feature = self.dropout(combined_feature)
        #combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return output#,p1,p2

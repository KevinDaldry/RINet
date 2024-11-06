# -*- coding:utf8 -*-

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from .darknet import *
from .modulation import RegionalIndicationGenerator, ComprehensiveAlignmentModule


def generate_coord(batch_size, height, width, gsd):
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    #print(('gsd', gsd), flush=True)
    gsd=gsd.view(batch_size, 1).repeat(1, height*width).reshape(-1, height, width).unsqueeze(1)
    h_gsd = gsd*(1./height)
    w_gsd = gsd*(1./width)
    #print(w_gsd, flush=True)
    
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    coord = torch.cat((coord, h_gsd, w_gsd), dim=1)
    return coord


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


class charEmbedding(nn.Module):
    def __init__(self, vocab_size, char_embedding_dim, model='gru', n_layers=1):
        super(charEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_embedding_dim)
        if model == 'gru':
            self.rnn = nn.GRU(char_embedding_dim, char_embedding_dim, n_layers, batch_first=True, bidirectional=True) # 
        elif model == 'lstm':
            self.rnn = nn.LSTM(char_embedding_dim, char_embedding_dim, n_layers, batch_first=True, bidirectional=True) #
        
        self.fc = nn.Linear(char_embedding_dim * 2, char_embedding_dim)

    def forward(self, input):
        x = self.char_embedding(input)
        output, _ = self.rnn(x)
        x = self.fc(output[:, -1, :])
        return x
        #return output[:, -1, :]


class RINet(nn.Module):
    def __init__(self, emb_size=512, jemb_drop_out=0.1, bert_model='bert-base-uncased', yolo_model='rsvg', leaky=False,
                 vocab_size=18, NFilm=3, fusion='prod', intmd=False, mstage=False, convlstm=False):
        super(RINet, self).__init__()
        self.emb_size = emb_size
        self.NFilm = NFilm
        self.intmd = intmd
        self.mstage = mstage
        self.convlstm = convlstm
        if bert_model == 'bert-base-uncased':
            self.textdim = 768
        else:
            self.textdim = 1024
        self.visudim = 256

        ## Visual model
        self.visumodel = Darknet(config_path=f'./model/yolov3_{yolo_model}.cfg', attn=False)
        self.visumodel.load_weights('./saved_models/yolov3.weights')

        ## Text model
        self.textmodel = BertModel.from_pretrained(
            r'/home/caizhicheng/xusiqi/rsvg_public/bert_source/bert-base-uncased-model')

        ## Mapping module
        self.mapping_visu = ConvBatchNormReLU(self.visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=True)
        self.mapping_visu_ast = ConvBatchNormReLU(self.visudim*4, emb_size, 1, 1, 0, 1, leaky=leaky, instance=True)

        self.mapping_lang = torch.nn.Sequential(
            nn.Linear(self.textdim, emb_size),
            nn.ReLU(inplace=True),
            nn.Dropout(jemb_drop_out),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(inplace=True), )

        self.rig = RegionalIndicationGenerator(self.emb_size)
        self.film = ComprehensiveAlignmentModule(NFilm=NFilm, textdim=emb_size, emb_size=emb_size, fusion=fusion,
                                                 intmd=(intmd or mstage or convlstm))
        self.char_embedding = charEmbedding(vocab_size=vocab_size, char_embedding_dim=emb_size, model='lstm')
        output_emb = emb_size
        if self.mstage:
            self.fcn_out = nn.ModuleDict()
            modules = OrderedDict()
            for n in range(0, NFilm):
                modules["out%d" % n] = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb, output_emb // 2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb // 2, 9 * 5, kernel_size=1))
            self.fcn_out.update(modules)
        else:
            if self.intmd:
                output_emb = emb_size * NFilm
            if self.convlstm:
                output_emb = emb_size
                self.global_out = ConvLSTM(input_size=(32, 32),
                                           input_dim=emb_size,
                                           hidden_dim=[emb_size],
                                           kernel_size=(1, 1),
                                           num_layers=1,
                                           batch_first=True,
                                           bias=True,
                                           return_all_layers=False)
            self.fcn_out = torch.nn.Sequential(
                ConvBatchNormReLU(output_emb, output_emb // 2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(output_emb // 2, 9 * 5, kernel_size=1))

    def forward(self, image, ori_image_gsds, word_postagger, word_id, word_mask, digitals):
        tokens_postagger_refobj, tokens_postagger_dist, _ = word_postagger[:, :, 0], word_postagger[:, :,
                                                                                     1], word_postagger[:, :, 2]

        # Language Module
        all_encoder_layers, _ = self.textmodel(word_id, \
                                               token_type_ids=None, attention_mask=word_mask)

        # Sentence feature at the first position [cls]
        raw_fword = (all_encoder_layers[-1] + all_encoder_layers[-2] \
                     + all_encoder_layers[-3] + all_encoder_layers[-4]) / 4
        raw_fword = raw_fword.detach()

        fword = Variable(torch.zeros(raw_fword.shape[0], raw_fword.shape[1], self.emb_size).cuda())
        fword_digital = Variable(torch.zeros(raw_fword.shape[0], raw_fword.shape[1], self.emb_size).cuda())

        batch_size, _, digital_count = digitals.shape
        for ii in range(raw_fword.shape[0]):
            ntoken = (word_mask[ii] != 0).sum()
            fword[ii, :ntoken, :] = F.normalize(self.mapping_lang(raw_fword[ii, :ntoken, :]), p=2, dim=1)

            digitals_idx = digitals[ii].sum(axis=1) != 0
            fword_digital[ii, digitals_idx] = F.normalize(self.char_embedding(digitals[ii, digitals_idx]), p=2, dim=1)

        raw_fword = fword  # shape [batch, 80, 512]
        early_raw_fword = raw_fword
        text_src, text_mask = early_raw_fword, word_mask

        # Visual Module

        raw_fvisu = self.visumodel(image)  # [batch, 1024, 32, 32] [batch, 512, 64, 64] [batch, 256, 128, 128]
        fvisu = raw_fvisu[2]  # torch.Size([4, 256, 128, 128])
        fvisu_ast = raw_fvisu[0]
        fvisu = self.mapping_visu(fvisu)  # torch.Size([4, 256, 128, 128])
        fvisu_ast = self.mapping_visu_ast(fvisu_ast)
        raw_fvisu = F.normalize(fvisu, p=2, dim=1)  # normalize each feature map
        fvisu_ast = F.normalize(fvisu_ast, p=2, dim=1)

        fvisu_ast = self.rig(early_raw_fword, fvisu_ast)

        coord = generate_coord(batch_size, raw_fvisu.size(2), raw_fvisu.size(3), ori_image_gsds)
        x, attnscore_list = self.film(raw_fvisu, early_raw_fword, coord, fvisu_ast, fsent=None, word_mask=word_mask)

        if self.mstage:
            outbox = []
            for film_ii in range(len(x)):
                outbox.append(self.fcn_out["out%d" % film_ii](x[film_ii]))
        elif self.convlstm:
            x = torch.stack(x, dim=1)
            output, state = self.global_out(x)
            output, hidden, cell = output[-1], state[-1][0], state[-1][1]
            outbox = self.fcn_out(hidden)
        else:
            outbox = []
            y = torch.stack(x, dim=1).view(batch_size, -1, raw_fvisu.size(2), raw_fvisu.size(3))
            outbox.append(self.fcn_out(y))
        outbox.insert(0, indication)

        return outbox, attnscore_list  ## list of (B,N,H,W)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
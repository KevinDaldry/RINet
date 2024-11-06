from collections import OrderedDict, Counter
import math
import random
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models
from torch.nn.init import kaiming_normal, kaiming_uniform
from .darknet import ConvBatchNormReLU, ConvBatchNormReLU_3d


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init_params(m.weight)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        # gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        # betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class RegionalIndicationGenerator(nn.Module):

    def __init__(self, in_channels):
        super(RegionalIndicationGenerator, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, in_channels // 2),
                                 nn.Linear(in_channels // 2, in_channels // 2),
                                 nn.Linear(in_channels // 2, 1))

    def forward(self, txt, img):
        B, C, H, W = img.shape
        img = img.view(B, C, -1)
        tnr = torch.matmul(txt, img).permute(0, 2, 1)
        out = self.mlp(tnr).view(B, H, W)
        return out


class Fusion(nn.Module):
    def __init__(self, emb_size):
        super(Fusion, self).__init__()
        self.emb_size = emb_size
        self.q = ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1)
        self.k = nn.Linear(emb_size, emb_size)
        self.v = nn.Linear(emb_size, emb_size)


    def forward(self, fvisu, fword, ast):
        B, Dvisu, H, W = fvisu.size()
        weight = F.interpolate(ast, scale_factor=4, mode='bilinear')
        ## second transformer fusion
        q = self.q(fvisu) * weight + fvisu
        k = self.k(fword)
        v = self.v(fword).unsqueeze(dim=-1)
        qkv = q * k + v
        return qkv


class CorrectionGate(nn.Module):
    def __init__(self, emb_dim=256):
        super(CorrectionGate, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.BatchNorm2d(emb_dim),
            nn.Conv2d(emb_dim, 1, 1),
            nn.Tanh())

    def forward(self, fvisu, ast):
        fvisu_ = self.pool(fvisu)
        weight = self.proj(fvisu_)
        ast_ = weight * fvisu_ + ast
        return ast_


def mask_softmax(attn_score, word_mask, tempuature=10., clssep=False, lstm=False):
    if len(attn_score.shape)!=2:
        attn_score = attn_score.squeeze(2).squeeze(2)
    word_mask_cp = word_mask[:,:attn_score.shape[1]].clone()
    score = F.softmax(attn_score*tempuature, dim=1)
    if not clssep:
        for ii in range(word_mask_cp.shape[0]):
            if lstm:
                word_mask_cp[ii,word_mask_cp[ii,:].sum()-1]=0
            else:
                word_mask_cp[ii,0]=0
                word_mask_cp[ii,word_mask_cp[ii,:].sum()]=0 ## set one to 0 already
    mask_score = score * word_mask_cp.float()
    mask_score = mask_score/(mask_score.sum(1)+1e-8).view(mask_score.size(0), 1).expand(mask_score.size(0), mask_score.size(1))
    return mask_score


class WordContributionLearner(nn.Module):
    def __init__(self, with_residual=True, textdim=768, visudim=512, emb_size=512, fusion='prod', cont_map=False,
                 lstm=False, baseline=False):
        super(WordContributionLearner, self).__init__()

        self.cont_map = cont_map    ## mapping context with language feature
        self.lstm = lstm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.fusion = fusion
        self.baseline = baseline

        if self.cont_map:
            self.sent_map = nn.Linear(768, emb_size)
            self.context_map = nn.Linear(emb_size, emb_size)
        if self.fusion == 'cat':
            self.attn_map = nn.Conv1d(textdim+visudim, emb_size//2, kernel_size=1)
        elif self.fusion == 'prod':
            assert(textdim==visudim) ## if product fusion
            self.attn_map = nn.Conv1d(visudim, emb_size//2, kernel_size=1)

        self.attn_score = nn.Conv1d(emb_size//2, 1, kernel_size=1)
        init_modules(self.modules())

    def forward(self, fvisu, fword, fvisu_ast, context_score, word_mask=None):
        fword = fword.permute(0, 2, 1)
        B, Dvisu, H, W = fvisu.size()
        B, Dlang, N = fword.size()
        B, N = context_score.size()
        assert Dvisu == Dlang

        ## word attention
        fvisu_ast_weight = F.interpolate(fvisu_ast, scale_factor=4, mode='bilinear')
        fvisu = fvisu * fvisu_ast_weight
        tile_visu = torch.mean(fvisu.view(B, Dvisu, -1), dim=2, keepdim=True).repeat(1, 1, N)
        if self.fusion == 'cat':
            context_tile = torch.cat([tile_visu,\
                fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)], dim=1)
        elif self.fusion == 'prod':
            context_tile = tile_visu * \
                fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)

        attn_feat = F.tanh(self.attn_map(context_tile))
        attn_score = self.attn_score(attn_feat).squeeze(1)
        mask_score = mask_softmax(attn_score, word_mask, lstm=self.lstm)
        attn_lang = torch.matmul(mask_score.view(B, 1, N), fword.permute(0, 2, 1))
        attn_lang = attn_lang.view(B, Dlang).squeeze(1)
        return attn_lang, attn_score


class ComprehensiveAlignmentModule(nn.Module):
    def __init__(self, NFilm=3, with_residual=True, textdim=768, emb_size=512, fusion='cat',
                 intmd=False, lstm=False, erasing=0.):
        super(ComprehensiveAlignmentModule, self).__init__()

        self.NFilm = NFilm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.cont_size = emb_size
        self.fusion = fusion
        self.intmd = intmd
        self.lstm = lstm
        self.erasing = erasing
        if self.fusion == 'cat':
            self.cont_size = emb_size*2

        self.modulesdict = nn.ModuleDict()
        modules = OrderedDict()
        modules["film0"] = WordContributionLearner(textdim=textdim, visudim=emb_size, emb_size=emb_size,
                                                   fusion=fusion, lstm=self.lstm)

        modules["sotf0"] = Fusion(emb_size=emb_size)
        modules["sotr0"] = CorrectionGate()
        for n in range(1, self.NFilm):
            modules["conv%d" % n] = ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1)
            modules["film%d" % n] = WordContributionLearner(textdim=textdim, visudim=emb_size, emb_size=emb_size,
                                                            fusion=fusion, lstm=self.lstm)
            modules["sotf0"] = Fusion(emb_size=emb_size)
            if n != self.NFilm - 1:
                modules["sotr%d" % n] = CorrectionGate()
        self.modulesdict.update(modules)

    def forward(self, fvisu, fword, fcoord, fvisu_ast, fsent=None, word_mask=None):
        B, N, Dlang = fword.size()
        intmd_feat, attnscore_list = [], []

        attn_lang, attn_score = self.modulesdict["film0"](fvisu, fword, Variable(torch.ones(B, N).cuda()),
                                                             fcoord, fsent=fsent, word_mask=word_mask)
        x = self.modulesdict["sotf0"](fvisu, fword, ast)
        fvisu_ast = self.modulesdict["sotr0"](x, fvisu_ast)
        attnscore_list.append(attn_score.view(B, N, 1, 1))
        if self.intmd:
            intmd_feat.append(x)
        if self.NFilm == 1:
            intmd_feat = [x]
        for n in range(1, self.NFilm):
            score_list = [mask_softmax(score.squeeze(2).squeeze(2), word_mask, lstm=self.lstm) for score in attnscore_list]
            score = torch.clamp(torch.max(torch.stack(score_list, dim=1), dim=1, keepdim=False)[0], min=0., max=1.)
            x = self.modulesdict["conv%d" % n](x)
            attn_lang, attn_score = self.modulesdict["film%d" % n](x, fword, (1-score), fcoord, fsent=fsent, word_mask=word_mask)
            x = self.modulesdict["sotf%d" % n](fvisu, fword, ast)
            if n != self.NFilm - 1:
                fvisu_ast = self.modulesdict["sotr%d" % n](x, fvisu_ast)
            attnscore_list.append(attn_score.view(B, N, 1, 1)) ## format match div loss in main func
            if self.intmd:
                intmd_feat.append(x)
            elif n == self.NFilm - 1:
                intmd_feat = [x]
        return intmd_feat, attnscore_list


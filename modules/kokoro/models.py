# Ported from https://huggingface.co/spaces/hexgrad/Kokoro-TTS/blob/main/models.py
# https://github.com/yl4579/StyleTTS2/blob/main/models.py

from munch import Munch
from torch.nn.utils import weight_norm, spectral_norm
import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from plbert import load_plbert
from istftnet import Decoder


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError(
                'Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in,
                                           output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in,
                                           output_padding=1, padding=1)
        else:
            raise RuntimeError(
                'Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                'Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError(
                'Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / np.sqrt(2)  # unit variance


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x)
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features


class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p

        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)

        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / np.sqrt(2)  # unit variance


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)

        x.masked_fill_(m, 0.0)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / np.sqrt(2)
        return out


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()

        self.text_encoder = DurationEncoder(sty_dim=style_dim,
                                            d_model=d_hid,
                                            nlayers=nlayers,
                                            dropout=dropout)

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        batch_size = d.shape[0]
        text_size = d.shape[1]

        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))

        en = (d.transpose(-1, -2) @ alignment)

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim,
                                      d_model // 2,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=True,
                                      dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)

        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)

        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)

                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)

        return x.transpose(-1, -2)

    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * np.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


# https://github.com/yl4579/StyleTTS2/blob/main/utils.py
def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def build_model(args, device):
    args = recursive_munch(args)
    assert args.decoder.type == 'istftnet', 'Decoder type unknown'
    decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                      resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
                      upsample_rates=args.decoder.upsample_rates,
                      upsample_initial_channel=args.decoder.upsample_initial_channel,
                      resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                      upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
                      gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size)
    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer,
                                 max_dur=args.max_dur, dropout=args.dropout)
    bert = load_plbert()
    bert_encoder = nn.Linear(bert.config.hidden_size, args.hidden_dim)
    for parent in [bert, bert_encoder, predictor, decoder, text_encoder]:
        for child in parent.children():
            if isinstance(child, nn.RNNBase):
                child.flatten_parameters()
    model = Munch(
        bert=bert.to(device).eval(),
        bert_encoder=bert_encoder.to(device).eval(),
        predictor=predictor.to(device).eval(),
        decoder=decoder.to(device).eval(),
        text_encoder=text_encoder.to(device).eval(),
    )
    return model


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


@torch.no_grad()
def forward(model, tokens, ref_s, speed):
    device = ref_s.device
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
        c_frame += pred_dur[0,i].item()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()
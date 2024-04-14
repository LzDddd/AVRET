# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch import Tensor
from signjoey.lstm import BiLSTMLayer
from signjoey.helpers import unpadding_clip_partition, padding_clip_reverse, unpadding_mask_partition
from typing import Callable, Optional, List
import random


class AdaptiveFusion(nn.Module):
    """
    adaptive Fusion Mechanism
    """

    def __init__(self, input_size_1=512, input_size_2=512, output_siz=2, bias=False):
        """
        adaptive Fusion instead of normal add
        :param input_size_1:
        :param input_size_2:
        :param output_siz:
        :param bias:
        """
        super(AdaptiveFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.weight_input_1 = nn.Linear(input_size_1, output_siz, bias=bias)
        self.weight_input_2 = nn.Linear(input_size_2, output_siz, bias=bias)
        self.layer_norm = nn.LayerNorm(input_size_1, eps=1e-5)   # 1e-5

    def forward(self, input_1, input_2):
        fm_sigmoid = self.sigmoid(self.weight_input_1(input_1) + self.weight_input_2(input_2))
        lambda1 = fm_sigmoid.clone().detach()[:, :, 0].unsqueeze(-1)
        lambda2 = fm_sigmoid.clone().detach()[:, :, 1].unsqueeze(-1)

        fused_output = input_1 + input_2 + torch.mul(lambda1, input_1) + torch.mul(lambda2, input_2)
        fused_output = self.layer_norm(fused_output)
        return fused_output


class AdaptiveMask(nn.Module):
    """
    Adaptive Masking Mechanism v3
    """

    def __init__(self, input_size=512, output_size=512, dropout=0.1):
        """
        AF module
        :param input_size: dimensionality of the input.
        :param output_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(AdaptiveMask, self).__init__()
        self.lstm = BiLSTMLayer(input_size=input_size, hidden_size=output_size, dropout=dropout)
        self.linear = nn.Linear(output_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-5)  # eps=1e-5

    def forward(self, input_tensor, input_len, k=20, mask=None):
        lstm_o = self.lstm(input_tensor, input_len, input_tensor.shape[1])  # B, T, D
        list_out = self.softmax(self.linear(lstm_o).squeeze(-1))
        values, indices = list_out.topk(k, dim=-1, largest=False, sorted=False)
        lstm_o = self.layer_norm(lstm_o)

        # update mask
        if mask is not None:
            sgn_mask_copy = mask.clone().detach()
            for b in range(input_tensor.shape[0]):
                sgn_mask_copy[b, :, indices[b]] = False
            return indices, lstm_o, sgn_mask_copy
        else:
            return indices, lstm_o


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1, use_gfm=False):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
            self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1, batch_size: int = 32
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout, use_gfm=False
        ) 
        self.dropout = nn.Dropout(dropout)
        self.size = size

        # LCSA
        self.clips_layer_norm = nn.ModuleList([nn.LayerNorm(size, eps=1e-6) for _ in range(batch_size)])
        self.clips_mha_blocks = nn.ModuleList([MultiHeadedAttention(num_heads, size, dropout) for _ in range(batch_size)])
        self.clips_feed_forward = nn.ModuleList(
            [
                PositionwiseFeedForward(input_size=size, ff_size=ff_size, dropout=dropout, use_gfm=False)
                for _ in range(batch_size)
            ]
        )
        self.clips_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(batch_size)])

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor, src_length=None) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """

        x_clips, clips_start_index, clips_end_index = unpadding_clip_partition(
            input_tensor=x,
            mask=mask,
            clip_size=16,  # 16
            overlap_area_size=3,  # 3
            src_length=src_length,
            extension_frames_size=4,  # 4
        )  # b * [clip_num, clip_size, dim_size]
        part_mask = unpadding_mask_partition(mask, clips_start_index, clips_end_index)

        x_clips_out = []
        for i, (clips, clips_layer, clips_mha, clips_ff, clips_drop, clips_mask) in enumerate(
                zip(
                    x_clips,
                    self.clips_layer_norm,
                    self.clips_mha_blocks,
                    self.clips_feed_forward,
                    self.clips_dropout,
                    part_mask
                )
        ):
            if clips.shape[1] == 16:  # FSDT
                clip_one, clips_two = clips[:, :, :128], clips[:, :, 128:]
                clips_two = clips_two.view(clips_two.shape[0], 16, 24, 16).permute(0, 3, 2, 1).contiguous()
                clips_two = clips_two.reshape(clips_two.shape[0], 16, -1).contiguous()
                clips = torch.cat([clip_one, clips_two], dim=-1)
            elif clips.shape[1] > 16:
                clip_one, clips_two = clips[:, :16, :128], clips[:, :16, 128:]
                clips_two = clips_two.view(clips_two.shape[0], 16, 24, 16).permute(0, 3, 2, 1).contiguous()
                clips_two = clips_two.reshape(clips_two.shape[0], 16, -1).contiguous()
                temp_clips = torch.cat([clip_one, clips_two], dim=-1)
                clips = torch.cat([temp_clips, clips[:, 16:, :]], dim=1)
            clips_h = clips_mha(clips_layer(clips), clips_layer(clips), clips_layer(clips), clips_mask)
            x_clips_out.append(clips_ff(clips_drop(clips_h) + clips))

        x_clips_out = padding_clip_reverse(
            original_tensor=x,
            clipped_tensor=x_clips_out,
            max_sentence_len=x.shape[1],
            clips_start_index=clips_start_index,
            clips_end_index=clips_end_index,
            overlap_area_size=3,  # 3
            padding_type='zero',
            is_mask=False
        )

        # MSA
        x_norm = self.layer_norm(x_clips_out)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)

        return o


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
            self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout, use_gfm=False
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
            self,
            x: Tensor = None,
            memory: Tensor = None,
            src_mask: Tensor = None,
            trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(k=memory, v=memory, q=h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o


class TransformerEncoderLayerV2(nn.Module):
    """
    One Transformer encoder layer has a LCSA and Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
            self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayerV2, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout, use_gfm=False
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

        # LCSA layers
        self.clips_layer_norm = nn.LayerNorm(16*self.size, eps=1e-6)
        self.clips_linear = nn.Linear(16*self.size, size)
        self.clips_mha_blocks = MultiHeadedAttention(num_heads, size, dropout)
        self.clips_feed_forward = PositionwiseFeedForward(input_size=size, ff_size=ff_size, dropout=dropout, use_gfm=False)
        self.clips_dropout = nn.Dropout(dropout)
        self.clips_reverse_linear = nn.Linear(size, 8192)

    def unpadding_clip_partition_v2(
        self,
        input_tensor: torch.Tensor,
        clip_size: int = 12,
        overlap_area_size: int = 3,
        src_length: torch.Tensor = None,
        extension_frames_size: int = 4
    ):
        """
        delete the padding of input tensor，and then get the partition clips from [B, M, D] tensor
        :param input_tensor: [B, M, D]
        :param mask: [B, 1, D]
        :param clip_size:
        :param overlap_area_size:
        :param src_length:
        :param extension_frames_size:
        :return:
        """
        assert clip_size > overlap_area_size and clip_size <= 16
        unbind_sign = input_tensor.unbind(0)
        src_length = src_length.int().tolist()
        ori_input_len = src_length

        # get the clip start and end index from original video
        clips_end_index = []
        clips_start_index = []
        for sentence_len in ori_input_len:
            temp_clips_end_index = []
            temp_clips_start_index = []

            if sentence_len <= clip_size:
                temp_clips_end_index.append(sentence_len)
                temp_clips_start_index.append(0)
            if clip_size < sentence_len <= clip_size + overlap_area_size:
                temp_clips_end_index.append(clip_size)
                temp_clips_start_index.append(0)

                temp_clips_end_index.append(sentence_len)
                temp_clips_start_index.append(sentence_len - clip_size)
            if sentence_len > clip_size + overlap_area_size:
                for ceidx in range(clip_size, sentence_len + 1, clip_size - overlap_area_size):
                    temp_clips_end_index.append(ceidx)
                    temp_clips_start_index.append(ceidx - clip_size)
                last_end_index = temp_clips_end_index[-1]
                if sentence_len - last_end_index > 0:
                    temp_clips_end_index.append(sentence_len)
                    temp_clips_start_index.append(sentence_len - clip_size)

            clips_end_index.append(temp_clips_end_index)
            clips_start_index.append(temp_clips_start_index)

        # get clips
        clips = []
        pad_random_index = []
        real_pad_idx = []
        for i, temp_sign in enumerate(unbind_sign):
            temp_clips = []
            for (sidx, eidx) in zip(clips_start_index[i], clips_end_index[i]):
                temp_clips.append(temp_sign[sidx: eidx, :])
            if len(temp_clips) == 1 and temp_clips[0].shape[0] < clip_size:
                diff = clip_size - temp_clips[0].shape[0]
                clip_tp_random_index = sorted(random.sample(range(temp_clips[0].shape[0]), diff))
                pad_random_index.append(clip_tp_random_index)
                tem_cp = []
                ttt = []
                for rdx in range(temp_clips[0].shape[0]):
                    tem_cp.append(temp_clips[0][rdx, :])
                    if rdx in clip_tp_random_index:
                        tem_cp.append(temp_clips[0][rdx, :])
                        ttt.append(len(tem_cp) - 1)
                real_pad_idx.append(ttt)
                temp_clips = [torch.stack(tem_cp, dim=0)]
            else:
                pad_random_index.append(None)
                real_pad_idx.append(None)
            clips.append(torch.stack(temp_clips, 0))

        # extension frames content
        for i, clip in enumerate(clips):
            if clip.shape[0] > 1:
                for clip_no in range(clip.shape[0]):
                    if clip_no == 0:  # 第一个片段
                        if clip.shape[0] <= extension_frames_size:
                            real_forward_extension_clip_nums = clip.shape[0] - 1
                        else:
                            real_forward_extension_clip_nums = extension_frames_size

                        front_each_clip_frames = []
                        for ex_no in range(real_forward_extension_clip_nums):
                            temp_ex_t = clip[clip_no + ex_no + 1, -1, :]
                            front_each_clip_frames.append(temp_ex_t)

                        for tpp in front_each_clip_frames:
                            clip[clip_no, -1, :] = clip[clip_no, -1, :] + tpp
                    elif clip_no == clip.shape[0] - 1:  # 最后一个片段
                        if clip.shape[0] <= extension_frames_size:
                            real_back_extension_clip_nums = clip.shape[0] - 1
                        else:
                            real_back_extension_clip_nums = extension_frames_size

                        back_each_clip_frames = []
                        for ex_no in range(real_back_extension_clip_nums):
                            temp_ex_t = clip[clip_no - 1 - ex_no, 0, :]
                            back_each_clip_frames.append(temp_ex_t)
                        for btp in back_each_clip_frames:
                            clip[clip_no, 0, :] = clip[clip_no, 0, :] + btp
                    else:  # 中间片段
                        # 向后扩展
                        if clip_no - 0 <= (extension_frames_size // 2):
                            real_back_extension_clip_nums = clip_no - 0
                        else:
                            real_back_extension_clip_nums = extension_frames_size // 2
                        back_each_clip_frames = []
                        for ex_no in range(real_back_extension_clip_nums):
                            temp_ex_t = clip[clip_no - 1 - ex_no, 0, :]
                            back_each_clip_frames.append(temp_ex_t)
                        for btp in back_each_clip_frames:
                            clip[clip_no, 0, :] = clip[clip_no, 0, :] + btp

                        # 向前扩展
                        if clip_no + (extension_frames_size // 2) <= clip.shape[0] - 1:
                            real_forward_extension_clip_nums = extension_frames_size // 2
                        else:
                            real_forward_extension_clip_nums = clip.shape[0] - 1 - clip_no
                        forward_each_clip_frames = []
                        for ex_no in range(real_forward_extension_clip_nums):
                            temp_ex_t = clip[clip_no + 1 + ex_no, -1, :]
                            forward_each_clip_frames.append(temp_ex_t)
                        for ftp in forward_each_clip_frames:
                            clip[clip_no, -1, :] = clip[clip_no, -1, :] + ftp

            # FSDT
            clip_one, clip_two = clip[:, :, :128], clip[:, :, 128:]
            clip_two = clip_two.view(clip_two.shape[0], clip_size, 24, clip_size).permute(0, 3, 2, 1).contiguous()
            clip_two = clip_two.reshape(clip_two.shape[0], clip_size, -1).contiguous()
            clips[i] = torch.cat([clip_one, clip_two], dim=-1)

        return clips, clips_start_index, clips_end_index, pad_random_index, real_pad_idx

    def padding_clip_reverse_v2(
        self,
        original_tensor,
        clipped_tensor,
        max_sentence_len: int = 475,
        clips_start_index: List = None,
        clips_end_index: List = None,
        overlap_area_size: int = 0,
        padding_type: str = 'zero',
        clip_size: int = 16,
        ori_pad_random_index=None,
        real_pad_idx=None
    ) -> torch.Tensor:
        """
        padding the clipped tensor and reverse
        :param original_tensor:
        :param clipped_tensor:
        :param max_sentence_len:
        :param clips_start_index:
        :param clips_end_index:
        :param overlap_area_size:
        :param padding_type:  zero or original
        :param clip_size:
        :return:
        """
        assert original_tensor.shape[0] == len(clipped_tensor) == len(ori_pad_random_index) == len(real_pad_idx)

        out = []
        for idx, clips in enumerate(clipped_tensor):  # [clip_num, clip_size, dim]
            temp_out = []
            for clip_idx in range(clips.shape[0]):
                if clip_idx == 0:
                    if ori_pad_random_index[idx] is None:
                        temp_out.append(clips[clip_idx, :, :])
                    else:
                        select_idx = []
                        for idw in range(clips.shape[1]):
                            if idw not in ori_pad_random_index[idx]:
                                select_idx.append(idw)
                        temp_out.append(clips[clip_idx, select_idx, :])
                elif clip_idx == clips.shape[0] - 1:
                    diff = clips_end_index[idx][-2] - clips_start_index[idx][-1]
                    temp_out.append(clips[clip_idx, diff:, :])
                else:
                    temp_out.append(clips[clip_idx, overlap_area_size:, :])

            temp_out = torch.cat(temp_out, 0)

            # padding
            if max_sentence_len > temp_out.shape[0]:
                if padding_type == 'zero':
                    pad = torch.zeros(max_sentence_len - temp_out.shape[0], original_tensor.shape[-1]).to(
                        temp_out.device)
                elif padding_type == 'original':
                    pad = original_tensor[idx, temp_out.shape[0]:, :].to(temp_out.device)
                else:
                    raise RuntimeError("unknown padding type !!!")
                temp_out = torch.cat([temp_out, pad], 0)

            out.append(temp_out)

        out = torch.stack(out, 0)
        return out

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor, src_length=None) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """

        x_clips, clips_start_index, clips_end_index, pad_random_index, real_pad_idx = self.unpadding_clip_partition_v2(
            input_tensor=x,
            clip_size=16,  # 16
            overlap_area_size=3,  # 3
            src_length=src_length,
            extension_frames_size=4,  # 4
        )  # b * [clip_num, clip_size, dim_size]
        gather_clips = [clip.reshape(clip.shape[0], -1).contiguous() for clip in x_clips]
        use_clips_num = [i.shape[0] for i in gather_clips]

        # pad x_clips
        pad_gather_clips = []
        for i in gather_clips:
            if (max(use_clips_num) - i.shape[0]) > 0:
                pad_gather_clips.append(
                    torch.cat([i, torch.zeros((max(use_clips_num) - i.shape[0], i.shape[1])).to(i.device)], dim=0))
            else:
                pad_gather_clips.append(i)
        pad_gather_clips = torch.stack(pad_gather_clips, dim=0)

        # get new mask
        new_mask = torch.zeros((mask.shape[0], mask.shape[1], max(use_clips_num)), dtype=torch.bool).to(mask.device)
        for idx, nm in enumerate(use_clips_num):
            new_mask[idx, :, :nm] = True

        # LCSA module
        clips_norm = self.clips_layer_norm(pad_gather_clips)   # 8192
        clips_norm = self.clips_linear(clips_norm)    # 512
        clips_h = self.clips_mha_blocks(clips_norm, clips_norm, clips_norm, new_mask)
        clips_output = self.clips_feed_forward(self.clips_dropout(clips_h) + clips_norm)

        # reverse
        clips_output = self.clips_reverse_linear(clips_output)
        clips_output = clips_output.unbind(0)
        ori_clips_lens = [i.shape[0] for i in x_clips]

        unbind_lcsa_o = []
        for idx, uclps in enumerate(clips_output):
            unbind_lcsa_o.append(uclps[:ori_clips_lens[idx], ...].view(-1, 16, self.size))
        clips_output = self.padding_clip_reverse_v2(
            original_tensor=x,
            clipped_tensor=unbind_lcsa_o,
            max_sentence_len=x.shape[1],
            clips_start_index=clips_start_index,
            clips_end_index=clips_end_index,
            overlap_area_size=3,  # 3
            padding_type='zero',
            ori_pad_random_index=pad_random_index,
            real_pad_idx=real_pad_idx
        )

        # MSA module
        x_norm = self.layer_norm(clips_output)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + clips_output
        o = self.feed_forward(h)

        return o

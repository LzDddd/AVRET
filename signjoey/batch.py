# coding: utf-8
import math
import random
import torch
import numpy as np
import gc


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        sgn_dim,
        is_train: bool = False,
        use_cuda: bool = False,
        frame_subsampling_ratio: int = None,
        random_frame_subsampling: bool = None,
        random_frame_masking_ratio: float = None,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        # Sign
        self.sgn, self.sgn_lengths = torch_batch.sgn
        self.sgn_lengths = self.sgn_lengths.int()
        # Here be dragons
        if frame_subsampling_ratio:
            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
            for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):
                features = features.clone()
                if random_frame_subsampling and is_train:
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data = features[: length.long(), :]
                tmp_data = tmp_data[init_frame::frame_subsampling_ratio]
                tmp_sgn[idx, 0: tmp_data.shape[0]] = tmp_data
                tmp_sgn_lengths[idx] = tmp_data.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths

        if is_train:
            if self.sgn.shape[1] < 475:  # 475
                temp_sgn = torch.zeros((self.sgn.shape[0], 475, sgn_dim)).to(self.sgn.device)
                for b in range(self.sgn.shape[0]):
                    temp_sgn[b, :self.sgn.shape[1], :] = self.sgn[b, :self.sgn.shape[1], :]
                self.sgn = temp_sgn
                del temp_sgn
                gc.collect()

        self.sgn_dim = sgn_dim
        self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt

            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if hasattr(torch_batch, "gls"):
            self.gls, self.gls_lengths = torch_batch.gls
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index

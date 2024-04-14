# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional, List
import numpy as np
import torch
from torch import nn, Tensor
from torchtext.data import Dataset
import yaml
from signjoey.vocabulary import GlossVocabulary, TextVocabulary


def clip_partition(
        input_tensor: torch.Tensor,
        sentence_len: int = 475,
        clip_size: int = 19,
        overlap_area_size: int = 4,
):
    """
    :param input_tensor: [B, M, D]
    :param sentence_len: all frames
    :param clip_size:
    :param overlap_area_size:
    :return:
    """
    clips_end_index = [i for i in range(clip_size, sentence_len, clip_size - overlap_area_size)]
    if sentence_len - clips_end_index[-1] > 0:
        clips_end_index.append(sentence_len)
    clips_start_index = [ceidx - clip_size for ceidx in clips_end_index]

    unbind_sign = input_tensor.unbind(0)
    clips = []
    for temp_sign in unbind_sign:
        temp_clips = []
        for (sidx, eidx) in zip(clips_start_index, clips_end_index):
            temp_clips.append(temp_sign[sidx: eidx, :])
        clips.append(torch.stack(temp_clips, 0))

    return clips, clips_start_index, clips_end_index


def clip_reverse(
        clipped_tensor,
        clips_start_index: List = None,
        clips_end_index: List = None,
        overlap_area_size: int = 0,
        is_mask: bool = False
) -> torch.Tensor:
    """
    :param clipped_tensor:
    :param clips_start_index:
    :param clips_end_index:
    :param overlap_area_size:
    :param is_mask:
    :return:
    """
    out = []
    for clips in clipped_tensor:  # [clip_num, clip_size, dim]
        temp_out = []
        for clip_idx in range(clips.shape[0]):
            if clip_idx == 0:
                temp_out.append(clips[clip_idx, :, :])
            elif clip_idx == clips.shape[0] - 1:
                diff = clips_end_index[-2] - clips_start_index[-1]
                if is_mask:
                    temp_out.append(clips[clip_idx, :, diff:])
                else:
                    temp_out.append(clips[clip_idx, diff:, :])
            else:
                if is_mask:
                    temp_out.append(clips[clip_idx, :, overlap_area_size:])
                else:
                    temp_out.append(clips[clip_idx, overlap_area_size:, :])
        if is_mask:
            out.append(torch.cat(temp_out, -1))
        else:
            out.append(torch.cat(temp_out, 0))
    out = torch.stack(out, 0).contiguous()

    return out


def mask_partition(
        mask: torch.Tensor,
        clips_start_index: List,
        clips_end_index: List,
        use_stack_mask: bool = False,
):
    """
    :param mask: [B, M]
    :param clips_start_index:
    :param clips_end_index:
    :param use_stack_mask:
    :return:
    """
    unbind_mask = mask.unbind(0)
    mask = []
    for temp_mask in unbind_mask:
        temp = []
        for (sidx, eidx) in zip(clips_start_index, clips_end_index):
            temp.append(temp_mask[:, sidx: eidx])
        mask.append(torch.stack(temp, 0))

    if use_stack_mask:
        mask = torch.stack(mask, dim=0).contiguous()
    del unbind_mask, temp

    return mask


def unpadding_clip_partition(
        input_tensor: torch.Tensor,
        mask: torch.Tensor = None,
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
    assert clip_size > overlap_area_size
    unbind_sign = input_tensor.unbind(0)
    src_length = src_length.int().tolist()
    ori_input_len = src_length

    # get the clip start and end index from original video
    clips_end_index = []
    clips_start_index = []
    for sentence_len in ori_input_len:
        temp_clips_end_index = []
        temp_clips_start_index = []
        if sentence_len - clip_size < overlap_area_size:
            temp_clips_end_index.append(sentence_len)
            temp_clips_start_index.append(0)
        else:
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
    for i, temp_sign in enumerate(unbind_sign):
        temp_clips = []
        for (sidx, eidx) in zip(clips_start_index[i], clips_end_index[i]):
            temp_clips.append(temp_sign[sidx: eidx, :])
        clips.append(torch.stack(temp_clips, 0))

    # extension frames content
    for i, clip in enumerate(clips):
        if clip.shape[0] == 1:
            break
        else:
            for clip_no in range(clip.shape[0]):
                if clip.shape[0] > 1:
                    if clip_no == 0:
                        if clip.shape[0] <= extension_frames_size:
                            real_forward_extension_clip_nums = clip.shape[0] - 1
                        else:
                            real_forward_extension_clip_nums = extension_frames_size

                        front_each_clip_frames = []
                        for ex_no in range(real_forward_extension_clip_nums):
                            temp_ex_t = clip[clip_no + ex_no + 1, -1, :]
                            front_each_clip_frames.append(temp_ex_t)

                        for tpp in front_each_clip_frames:
                            clips[i][clip_no, -1, :] = clips[i][clip_no, -1, :] + tpp
                    elif clip_no == clip.shape[0] - 1:
                        if clip.shape[0] <= extension_frames_size:
                            real_back_extension_clip_nums = clip.shape[0] - 1
                        else:
                            real_back_extension_clip_nums = extension_frames_size

                        back_each_clip_frames = []
                        for ex_no in range(real_back_extension_clip_nums):
                            temp_ex_t = clip[clip_no - 1 - ex_no, 0, :]
                            back_each_clip_frames.append(temp_ex_t)
                        for btp in back_each_clip_frames:
                            clips[i][clip_no, 0, :] = clips[i][clip_no, 0, :] + btp
                    else:
                        if clip_no - 0 <= (extension_frames_size // 2):
                            real_back_extension_clip_nums = clip_no - 0
                        else:
                            real_back_extension_clip_nums = extension_frames_size // 2
                        back_each_clip_frames = []
                        for ex_no in range(real_back_extension_clip_nums):
                            temp_ex_t = clip[clip_no - 1 - ex_no, 0, :]
                            back_each_clip_frames.append(temp_ex_t)
                        for btp in back_each_clip_frames:
                            clips[i][clip_no, 0, :] = clips[i][clip_no, 0, :] + btp

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
                            clips[i][clip_no, -1, :] = clips[i][clip_no, -1, :] + ftp

    return clips, clips_start_index, clips_end_index


def padding_clip_reverse(
        original_tensor,
        clipped_tensor,
        max_sentence_len: int = 475,
        clips_start_index: List = None,
        clips_end_index: List = None,
        overlap_area_size: int = 0,
        padding_type: str = 'zero',
        is_mask: bool = False
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
    :param is_mask:
    :return:
    """
    assert original_tensor.shape[0] == len(clipped_tensor)

    out = []
    for idx, clips in enumerate(clipped_tensor):  # [clip_num, clip_size, dim]
        temp_out = []
        for clip_idx in range(clips.shape[0]):
            if clip_idx == 0:
                temp_out.append(clips[clip_idx, :, :])
            elif clip_idx == clips.shape[0] - 1:
                diff = clips_end_index[idx][-2] - clips_start_index[idx][-1]
                if is_mask:
                    temp_out.append(clips[clip_idx, :, diff:])
                else:
                    temp_out.append(clips[clip_idx, diff:, :])
            else:
                if is_mask:
                    temp_out.append(clips[clip_idx, :, overlap_area_size:])
                else:
                    temp_out.append(clips[clip_idx, overlap_area_size:, :])

        if is_mask:
            temp_out = torch.cat(temp_out, -1)
        else:
            temp_out = torch.cat(temp_out, 0)

        # padding
        if is_mask:
            if max_sentence_len > temp_out.shape[-1]:
                assert padding_type == 'zero'
                pad = torch.zeros(original_tensor.shape[1], max_sentence_len - temp_out.shape[-1]).to(temp_out.device)
                temp_out = torch.cat([temp_out, pad], -1)
        else:
            if max_sentence_len > temp_out.shape[0]:
                if padding_type == 'zero':
                    pad = torch.zeros(max_sentence_len - temp_out.shape[0], original_tensor.shape[-1]).to(temp_out.device)
                elif padding_type == 'original':
                    pad = original_tensor[idx, temp_out.shape[0]:, :].to(temp_out.device)
                else:
                    raise RuntimeError("unknown padding type !!!")
                temp_out = torch.cat([temp_out, pad], 0)
        out.append(temp_out)

    out = torch.stack(out, 0)
    return out


def unpadding_mask_partition(
        mask: torch.Tensor,
        clips_start_index: List,
        clips_end_index: List,
):
    """
    get the mask clips
    :param mask: [B, M]
    :param clips_start_index:
    :param clips_end_index:
    :return:
    """
    unbind_mask = mask.unbind(0)
    # get mask clips
    mask = []
    for i, temp in enumerate(unbind_mask):
        temp_mask = []
        for (sidx, eidx) in zip(clips_start_index[i], clips_end_index[i]):
            temp_mask.append(temp[:, sidx: eidx])
        mask.append(torch.stack(temp_mask, 0))
    return mask


def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(
        train_data: Dataset,
        valid_data: Dataset,
        test_data: Dataset,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )

    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu")
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
            .transpose(0, 1)
            .repeat(count, 1)
            .transpose(0, 1)
            .contiguous()
            .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

# coding=utf-8

import os
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from transformers import BertModel, BertTokenizer, XLMModel, XLMTokenizer, RobertaModel, RobertaTokenizer, \
    XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
from train_utils import get_logger

import ot
from torchmetrics.functional import pairwise_cosine_similarity

LOG = get_logger(__name__)

def nxn_cos_sim(A, B, dim=-1, eps=1e-8):
    numerator = torch.bmm(A, torch.permute(B, (0, 2, 1))) 
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.bmm(A_l2.unsqueeze(-1), B_l2.unsqueeze(1))), torch.tensor(eps))   
    return torch.div(numerator, denominator)


def return_extended_attention_mask(attention_mask, dtype):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids or attention_mask"
        )
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask



class SentenceAligner_word(object):
    def __init__(self, args, model):

        self.guide = None
        self.softmax_threshold = args.softmax_threshold
        self.embed_loader = model

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, x.size(-1))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_subword_matrix(self, args, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID, output_prob=False):

        output_src,output_tgt = self.embed_loader(
            inputs_src=inputs_src, inputs_tgt=inputs_tgt, attention_mask_src=(inputs_src != PAD_ID),
            attention_mask_tgt=(inputs_tgt != PAD_ID), guide=None, align_layer=args.align_layer,
            extraction=args.extraction, softmax_threshold=args.softmax_threshold, do_infer=True,
        )

        align_matrix_all_layers = {}

        for layer_id in range(1, len(output_src.hidden_states)):

            hidden_states_src = output_src.hidden_states[layer_id]
            hidden_states_tgt = output_tgt.hidden_states[layer_id]
            # mask
            attention_mask_src = ((inputs_src == PAD_ID) + (inputs_src == CLS_ID) + (inputs_src == SEP_ID)).float()
            attention_mask_tgt = ((inputs_tgt == PAD_ID) + (inputs_tgt == CLS_ID) + (inputs_tgt == SEP_ID)).float()

            mask_src = 1 - attention_mask_src
            mask_tgt = 1 - attention_mask_tgt


            len_src = torch.sum(1 - attention_mask_src, -1)
            len_tgt = torch.sum(1 - attention_mask_tgt, -1)
            attention_mask_src = return_extended_attention_mask(1 - attention_mask_src, hidden_states_src.dtype)
            attention_mask_tgt = return_extended_attention_mask(1 - attention_mask_tgt, hidden_states_tgt.dtype)

            # qkv
            query_src = self.transpose_for_scores(hidden_states_src)
            query_tgt = self.transpose_for_scores(hidden_states_tgt)
            key_src = query_src
            key_tgt = query_tgt
            value_src = query_src
            value_tgt = query_tgt

            if args.extraction in ["balancedOT", "unbalancedOT", "partialOT"]:
                # Iterate through embeddings
                output = torch.full((query_src.size()[0], 1, query_src.size()[2], key_tgt.size()[2]), 0.0)
                for i, (source, target, source_mask, target_mask) in enumerate(zip(query_src, key_tgt, mask_src, mask_tgt)):

                    # Extract non-masked tokens
                    nomask_source = source[0][source_mask.nonzero()].squeeze(1)
                    nomask_target = target[0][target_mask.nonzero()].squeeze(1)
                
                    # Calculate cosine distance and normalize
                    cosine_distance = 1 - pairwise_cosine_similarity(nomask_source, nomask_target)
                    #cosine_sim = torch.nan_to_num(cosine_sim)
                    size = cosine_distance.size()
                    cosine_distance -= cosine_distance.min()
                    cosine_distance /= cosine_distance.max()

                    # Create initial distributions
                    source_distribution = torch.full((size[0],1), 1.0 / size[0]).squeeze(1)
                    target_distribution = torch.full((size[1],1), 1.0 / size[1]).squeeze(1)

                    reg = 0.1
                    reg_m = 0.1
                    if args.extraction == "balancedOT":
                        transition_matrix = ot.bregman.sinkhorn_log(source_distribution, target_distribution, cosine_distance, reg, numItermax = 250)
                    elif args.extraction == "unbalancedOT":
                        transition_matrix = ot.unbalanced.sinkhorn_unbalanced(source_distribution, target_distribution, cosine_distance, reg, reg_m)
                    elif args.extraction == "partialOT":
                        transition_matrix = ot.partial.entropic_partial_wasserstein(source_distribution, target_distribution, cosine_distance, reg)

                    output[i, 0, 1:size[0] + 1, 1:size[1] + 1] = transition_matrix

                if self.guide is None:
                    if args.extraction == "balancedOT":
                        align_matrix = output > 0.005
                    elif args.extraction == "unbalancedOT":
                        align_matrix = output > 0.01
                    elif args.extraction == "partialOT":
                        align_matrix = output > 0.01

                    if not output_prob:
                        # return align_matrix
                        align_matrix_all_layers[layer_id] = align_matrix
                    # A heuristic of generating the alignment probability
            elif args.extraction == 'softmax':                
                # att
                attention_scores = torch.matmul(query_src, key_tgt.transpose(-1, -2))
                attention_scores_src = attention_scores + attention_mask_tgt
                attention_scores_tgt = attention_scores + attention_mask_src.transpose(-1, -2)

                attention_probs_src = nn.Softmax(dim=-1)(
                    attention_scores_src)  # if extraction == 'softmax' else entmax15(attention_scores_src, dim=-1)
                attention_probs_tgt = nn.Softmax(dim=-2)(
                    attention_scores_tgt)  # if extraction == 'softmax' else entmax15(attention_scores_tgt, dim=-2)

                if self.guide is None:
                    # threshold = softmax_threshold if extraction == 'softmax' else 0
                    threshold = self.softmax_threshold
                    
                    align_matrix = (attention_probs_src > threshold) * (attention_probs_tgt > threshold)
                    if not output_prob:
                        # return align_matrix
                        align_matrix_all_layers[layer_id] = align_matrix
                    # A heuristic of generating the alignment probability
                    """
                    attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src/torch.sqrt(len_tgt.view(-1, 1, 1, 1)))
                    attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt/torch.sqrt(len_src.view(-1, 1, 1, 1)))
                    align_prob = (2*attention_probs_src*attention_probs_tgt)/(attention_probs_src+attention_probs_tgt+1e-9)
                    return align_matrix, align_prob
                    """

        return align_matrix_all_layers

    def get_aligned_word(self, args, inputs_src, inputs_tgt, bpe2word_map_src, bpe2word_map_tgt, PAD_ID, CLS_ID, SEP_ID,
                         output_prob=False):

        attention_probs_inter_all_layers = self.get_subword_matrix(args, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID,
                                                                   output_prob)
        if output_prob:
            attention_probs_inter, alignment_probs = attention_probs_inter
            alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]

        word_aligns_all_layers = {}

        for layer_id in attention_probs_inter_all_layers:

            attention_probs_inter = attention_probs_inter_all_layers[layer_id].float()

            word_aligns = []
            attention_probs_inter = attention_probs_inter[:, 0, 1:-1, 1:-1]

            for idx, (attention, b2w_src, b2w_tgt) in enumerate(
                    zip(attention_probs_inter, bpe2word_map_src, bpe2word_map_tgt)):
                aligns = set() if not output_prob else dict()
                non_zeros = torch.nonzero(attention)
                for i, j in non_zeros:
                    word_pair = (b2w_src[i], b2w_tgt[j])
                    if output_prob:
                        prob = alignment_probs[idx, i, j]
                        if not word_pair in aligns:
                            aligns[word_pair] = prob
                        else:
                            aligns[word_pair] = max(aligns[word_pair], prob)
                    else:
                        aligns.add(word_pair)
                word_aligns.append(aligns)

            word_aligns_all_layers[layer_id] = word_aligns
        return word_aligns_all_layers

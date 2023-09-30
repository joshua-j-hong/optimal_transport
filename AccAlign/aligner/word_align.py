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
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

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
        self.alignment_threshold = args.alignment_threshold
        self.embed_loader = model

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, x.size(-1))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_subword_matrix(self, args, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID, output_prob=False):

        inputs_src = inputs_src.to(args.device)
        inputs_tgt = inputs_tgt.to(args.device)

        output_src,output_tgt = self.embed_loader(
            inputs_src=inputs_src, inputs_tgt=inputs_tgt, attention_mask_src=(inputs_src != PAD_ID),
            attention_mask_tgt=(inputs_tgt != PAD_ID), guide=None, align_layer=args.align_layer,
            extraction=args.extraction, alignment_threshold=args.alignment_threshold, do_infer=True,
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
                output_source = torch.full((query_src.size()[0], 1, query_src.size()[2], key_tgt.size()[2]), 0.0)
                output_target = torch.full((query_src.size()[0], 1, query_src.size()[2], key_tgt.size()[2]), 0.0)

                # embed_mean = (torch.sum(query_src, (0,2)) + torch.sum(key_tgt, (0,2))) / (query_src.size()[0] * query_src.size()[2] + key_tgt.size()[0] * key_tgt.size()[2])
                # query_src = query_src - embed_mean
                # key_tgt = key_tgt - embed_mean


                for i, (source, target, source_mask, target_mask) in enumerate(zip(query_src, key_tgt, mask_src, mask_tgt)):

                    # Extract non-masked tokens
                    nomask_source = source[0][source_mask.nonzero()].squeeze(1)
                    nomask_target = target[0][target_mask.nonzero()].squeeze(1)
                
                    # Calculate cosine distance and normalize
                    eps = 1e-10
                    if args.cost_function == "cosine_sim":
                        cosine_similarity = (torch.matmul(torch.nn.functional.normalize(nomask_source), torch.nn.functional.normalize(nomask_target).t()) + 1.0) / 2

                        similarity = cosine_similarity

                        # Matrix normalize
                        cosine_min = cosine_similarity.min()
                        cosine_max = cosine_similarity.max()
                        cosine_similarity = (cosine_similarity - cosine_min + eps) / (cosine_max - cosine_min + eps)

                        distance = 1 - cosine_similarity
                        source_norm_distance = distance
                        target_norm_distance = distance

                        # Row/Column normalize
                        source_min = distance.min(1)[0].unsqueeze(1)
                        target_min = distance.min(0)[0].unsqueeze(0)
                        source_norm_distance = (distance - source_min + eps) / (distance.max(1)[0].unsqueeze(1) - source_min + eps)
                        target_norm_distance = (distance - target_min + eps) / (distance.max(0)[0].unsqueeze(0) - target_min + eps)
                        distance = source_norm_distance

                    elif args.cost_function == "euclidean_distance":
                        euclidean_distance = torch.cdist(nomask_source, nomask_target, p=2)
                        # Matrix normalize
                        # euclidean_min = euclidean_distance.min()
                        # euclidean_max = euclidean_distance.max()
                        # euclidean_distance = (euclidean_distance - euclidean_min + eps) / (euclidean_max - euclidean_min + eps)
                        # distance = euclidean_distance
                        # source_norm_distance = distance
                        # target_norm_distance = distance

                        # Row/Column normalize
                        source_min = euclidean_distance.min(1)[0].unsqueeze(1)
                        target_min = euclidean_distance.min(0)[0].unsqueeze(0)
                        source_norm_distance = (euclidean_distance - source_min + eps) / (euclidean_distance.max(1)[0].unsqueeze(1) - source_min + eps)
                        target_norm_distance = (euclidean_distance - target_min + eps) / (euclidean_distance.max(0)[0].unsqueeze(0) - target_min + eps)
                        distance = source_norm_distance

                    size = distance.size()

                    # Create initial distributions

                    if args.fertility_distribution == "uniform":
                        source_distribution = torch.full((size[0],1), 1.0 / size[0]).squeeze(1).to(args.device)
                        source_norms = source_distribution
                        target_distribution = torch.full((size[1],1), 1.0 / size[1]).squeeze(1).to(args.device)
                        target_norms = target_distribution
                    elif args.fertility_distribution == "l2_norm":
                        source_norms = torch.linalg.norm(nomask_source, dim=1)
                        source_distribution = source_norms /  torch.sum(source_norms)

                        target_norms = torch.linalg.norm(nomask_target, dim=1)
                        target_distribution = target_norms /  torch.sum(target_norms)

                        # source_norms = (cosine_similarity).sum(dim = -1)
                        # source_norms = source_norms - source_norms.min() + 1
                        # target_norms = (cosine_similarity).sum(dim = -2)
                        # target_norms = target_norms - target_norms.min() + 1

                        # torch.set_printoptions(sci_mode=False)
                        # print(cosine_similarity)
                        # print(source_norms)
                        # print(target_norms)

                    reg = args.entropy_regularization
                    reg_m = args.marginal_regularization
                    mass_transported = args.mass_transported
                    # Apply OT
                    if args.extraction == "balancedOT":
                        source_transition_matrix = ot.bregman.sinkhorn_log(source_distribution, target_distribution, source_norm_distance, reg, numItermax = 300)
                        target_transition_matrix = ot.bregman.sinkhorn_log(source_distribution, target_distribution, target_norm_distance, reg, numItermax = 300)
                    elif args.extraction == "unbalancedOT":
                        source_transition_matrix = ot.unbalanced.sinkhorn_unbalanced(source_norms, target_norms, source_norm_distance, reg, reg_m)
                        target_transition_matrix = ot.unbalanced.sinkhorn_unbalanced(source_norms, target_norms, target_norm_distance, reg, reg_m)
                    elif args.extraction == "partialOT":
                        m = mass_transported * torch.minimum(torch.sum(source_norms), torch.sum(target_norms))

                        source_transition_matrix = ot.partial.entropic_partial_wasserstein(source_norms, target_norms, source_norm_distance, reg, m)
                        target_transition_matrix = ot.partial.entropic_partial_wasserstein(source_norms, target_norms, target_norm_distance, reg, m)

                    transition_source = source_transition_matrix
                    transition_target = target_transition_matrix

                    # transition_source = torch.div(source_transition_matrix, source_norms.unsqueeze(1))
                    # transition_target = torch.div(target_transition_matrix, target_norms.unsqueeze(0))
                    
                    eps = 1e-10
                    matrix_min = transition_source.min()
                    matrix_max = transition_source.max()
                    transition_source = (transition_source - matrix_min + eps) / (matrix_max - matrix_min + eps)

                    matrix_min = transition_target.min()
                    matrix_max = transition_target.max()
                    transition_target = (transition_target - matrix_min + eps) / (matrix_max - matrix_min + eps)

                    output_source[i, 0, 1:size[0] + 1, 1:size[1] + 1] = transition_source
                    output_target[i, 0, 1:size[0] + 1, 1:size[1] + 1] = transition_target

                    # if layer_id == 8 and i == 0:
                    #     torch.set_printoptions(sci_mode=False)
                    #     for tensor in torch.minimum(transition_source, transition_target):
                    #         print(tensor)

                if self.guide is None:
                    align_matrix = (output_source > args.alignment_threshold) * (output_target > args.alignment_threshold)

                    if not output_prob:
                        # return align_matrix
                        align_matrix_all_layers[layer_id] = align_matrix
                    else:
                        align_matrix_all_layers[layer_id] = torch.minimum(output_source, output_target)
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
                    
                    align_matrix = (attention_probs_src > self.alignment_threshold) * (attention_probs_tgt > self.alignment_threshold)
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

    def get_aligned_word_from_matrix(self, args, inputs_src, inputs_tgt, matrix, bpe2word_map_src, bpe2word_map_tgt, PAD_ID, CLS_ID, SEP_ID,
                         threshold, output_prob=False):

        attention_probs_inter_all_layers = matrix
        if output_prob:
            attention_probs_inter, alignment_probs = attention_probs_inter
            alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]

        word_aligns_all_layers = {}

        for layer_id in attention_probs_inter_all_layers:

            attention_probs_inter = (attention_probs_inter_all_layers[layer_id] > threshold).float()

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

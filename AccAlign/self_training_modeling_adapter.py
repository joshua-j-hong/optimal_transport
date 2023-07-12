import transformers
from transformers import AutoModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from transformers import PreTrainedModel

import ot
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance


PAD_ID=0
CLS_ID=101
SEP_ID=102

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


class ModelGuideHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fnc = nn.BCELoss(reduction='sum')
        #self.loss_fnc = nn.MSELoss(reduction='sum')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, x.size(-1))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_src, hidden_states_tgt,
        inputs_src, inputs_tgt,
        guide=None,
        extraction='softmax', alignment_threshold=0.1,
        entropy_regularization = 0.1,
        marginal_regularization = 0.5,
        mass_transported = 1,
        fertility_distribution = 'l2_norm',
        cost_function = "cosine_sim",
        output_prob=False,
    ):
        #mask
        attention_mask_src = ( (inputs_src==PAD_ID) + (inputs_src==CLS_ID) + (inputs_src==SEP_ID) ).float()
        attention_mask_tgt = ( (inputs_tgt==PAD_ID) + (inputs_tgt==CLS_ID) + (inputs_tgt==SEP_ID) ).float()

        mask_src = 1 - attention_mask_src
        mask_tgt = 1 - attention_mask_tgt

        len_src = torch.sum(1-attention_mask_src, -1)
        len_tgt = torch.sum(1-attention_mask_tgt, -1)
        attention_mask_src = return_extended_attention_mask(1-attention_mask_src, hidden_states_src.dtype)
        attention_mask_tgt = return_extended_attention_mask(1-attention_mask_tgt, hidden_states_src.dtype)

        #qkv
        query_src = self.transpose_for_scores(hidden_states_src)
        query_tgt = self.transpose_for_scores(hidden_states_tgt)
        key_src = query_src
        key_tgt = query_tgt
        value_src = query_src
        value_tgt = query_tgt

        ###
        if extraction in ["balancedOT", "unbalancedOT", "partialOT"]:
            # Iterate through embeddings
            output_source = torch.full((query_src.size()[0], 1, query_src.size()[2], key_tgt.size()[2]), 0.0)
            output_target = torch.full((query_src.size()[0], 1, query_src.size()[2], key_tgt.size()[2]), 0.0)
            
            so_loss = 0
            
            for i, (source, target, source_mask, target_mask) in enumerate(zip(query_src, key_tgt, mask_src, mask_tgt)):

                # Extract non-masked tokens
                nomask_source = source[0][source_mask.nonzero()].squeeze(1)
                nomask_target = target[0][target_mask.nonzero()].squeeze(1)
            
                eps = 1e-10
                if cost_function == "cosine_sim":
                    cosine_similarity = (torch.matmul(torch.nn.functional.normalize(nomask_source), torch.nn.functional.normalize(nomask_target).t()) + 1.0) / 2
                    # Matrix normalize
                    # cosine_min = cosine_similarity.min()
                    # cosine_max = cosine_similarity.max()
                    # cosine_similarity = (cosine_similarity - cosine_min + eps) / (cosine_max - cosine_min + eps)
                    distance = 1 - cosine_similarity

                    # Row/Column normalize
                    source_min = distance.min(1)[0].unsqueeze(1)
                    target_min = distance.min(0)[0].unsqueeze(0)
                    source_norm_distance = (distance - source_min + eps) / (distance.max(1)[0].unsqueeze(1) - source_min + eps)
                    target_norm_distance = (distance - target_min + eps) / (distance.max(0)[0].unsqueeze(0) - target_min + eps)
                    distance = source_norm_distance

                elif cost_function == "euclidean_distance":
                    euclidean_distance = torch.cdist(nomask_source, nomask_target, p=2)
                    euclidean_min = euclidean_distance.min()
                    euclidean_max = euclidean_distance.max()
                    euclidean_distance = (euclidean_distance - euclidean_min + eps) / (euclidean_max - euclidean_min + eps)
                    distance = euclidean_distance

                size = distance.size()

                # Create initial distributions
                if fertility_distribution == "uniform":
                    source_distribution = torch.full((size[0],1), 1.0 / size[0]).squeeze(1)
                    target_distribution = torch.full((size[1],1), 1.0 / size[1]).squeeze(1)
                elif fertility_distribution == "l2_norm":
                    source_norms = torch.linalg.norm(nomask_source, dim=1)
                    source_distribution = source_norms /  torch.sum(source_norms)

                    target_norms = torch.linalg.norm(nomask_target, dim=1)
                    target_distribution = target_norms /  torch.sum(target_norms)


                reg = entropy_regularization
                reg_m = marginal_regularization
                if extraction == "balancedOT":
                    source_transition_matrix = ot.bregman.sinkhorn_log(source_distribution, target_distribution, source_norm_distance, reg, numItermax = 300)
                    target_transition_matrix = ot.bregman.sinkhorn_log(source_distribution, target_distribution, target_norm_distance, reg, numItermax = 300)
                elif extraction == "unbalancedOT":
                    source_transition_matrix = ot.unbalanced.sinkhorn_unbalanced(source_distribution, target_distribution, source_norm_distance, reg, reg_m)
                    target_transition_matrix = ot.unbalanced.sinkhorn_unbalanced(source_distribution, target_distribution, target_norm_distance, reg, reg_m)
                elif extraction == "partialOT":
                    m = mass_transported * torch.minimum(torch.sum(source_distribution), torch.sum(target_distribution))

                    source_transition_matrix = ot.partial.entropic_partial_wasserstein(source_distribution, target_distribution, source_norm_distance, reg, m)
                    target_transition_matrix = ot.partial.entropic_partial_wasserstein(source_distribution, target_distribution, target_norm_distance, reg, m)

                transition_source = source_transition_matrix 
                transition_target = target_transition_matrix

                eps = 1e-10
                matrix_min = transition_source.min()
                matrix_max = transition_source.max()
                transition_source = (transition_source - matrix_min + eps) / (matrix_max - matrix_min + eps)

                matrix_min = transition_target.min()
                matrix_max = transition_target.max()
                transition_target = (transition_target - matrix_min + eps) / (matrix_max - matrix_min + eps)

                output_source[i, 0, 1:size[0] + 1, 1:size[1] + 1] = transition_source
                output_target[i, 0, 1:size[0] + 1, 1:size[1] + 1] = transition_target


                #if guide is not None:
                    # so_loss += self.loss_fnc(transition_source, guide[i, 0, 1:size[0] + 1, 1:size[1] + 1])
                    # so_loss += self.loss_fnc(transition_target, guide[i, 0, 1:size[0] + 1, 1:size[1] + 1])
            
                    # so_loss += self.loss_fnc(transition_matrix, guide[i, 0, 1:size[0] + 1, 1:size[1] + 1])

                    # so_loss_src = torch.sum(torch.sum (transition_source*guide[i, 0, 1:size[0] + 1, 1:size[1] + 1], -1), -1).view(-1)
                    # so_loss_tgt = torch.sum(torch.sum (transition_target*guide[i, 0, 1:size[0] + 1, 1:size[1] + 1], -1), -1).view(-1)

                    # loss = so_loss_src/len_src + so_loss_tgt/len_tgt

                    # loss = torch.sum(torch.sum((1 - distance) *guide[i, 0, 1:size[0] + 1, 1:size[1] + 1], -1), -1).view(-1)
                    # so_loss = so_loss - torch.sum(loss)

                    #so_loss += self.loss_fnc((1 - distance), guide[i, 0, 1:size[0] + 1, 1:size[1] + 1])


            if guide is None:
                align_matrix = (output_source > alignment_threshold) * (output_target > alignment_threshold)
                if not output_prob:
                    return align_matrix
                # A heuristic of generating the alignment probability
                attention_probs_src = nn.Softmax(dim=-1)(align_matrix/torch.sqrt(len_tgt.view(-1, 1, 1, 1)))
                attention_probs_tgt = nn.Softmax(dim=-2)(align_matrix/torch.sqrt(len_src.view(-1, 1, 1, 1)))
                align_prob = (2*attention_probs_src*attention_probs_tgt)/(attention_probs_src+attention_probs_tgt+1e-9)
                return align_matrix, align_prob



            # so_loss_src = torch.sum(torch.sum (output_source*guide, -1), -1).view(-1)
            # so_loss_tgt = torch.sum(torch.sum (output_target*guide, -1), -1).view(-1)

            # so_loss = so_loss_src/len_src + so_loss_tgt/len_tgt
            # so_loss = -torch.mean(so_loss)

            so_loss += self.loss_fnc(output_source, guide)
            so_loss += self.loss_fnc(output_target, guide)

            #so_loss += self.loss_fnc(torch.minimum(output_source, output_target), guide)
            so_loss = so_loss / query_src.size()[0]

            return so_loss

        elif extraction == 'softmax':                
            # att
            attention_scores = torch.matmul(query_src, key_tgt.transpose(-1, -2))
            attention_scores_src = attention_scores + attention_mask_tgt
            attention_scores_tgt = attention_scores + attention_mask_src.transpose(-1, -2)

            attention_probs_src = nn.Softmax(dim=-1)(
                attention_scores_src)  # if extraction == 'softmax' else entmax15(attention_scores_src, dim=-1)
            attention_probs_tgt = nn.Softmax(dim=-2)(
                attention_scores_tgt)  # if extraction == 'softmax' else entmax15(attention_scores_tgt, dim=-2)

            if guide is None:
                # threshold = softmax_threshold if extraction == 'softmax' else 0
                
                align_matrix = (attention_probs_src > alignment_threshold) * (attention_probs_tgt > alignment_threshold)
                if not output_prob:
                    return align_matrix
                # A heuristic of generating the alignment probability
                attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src/torch.sqrt(len_tgt.view(-1, 1, 1, 1)))
                attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt/torch.sqrt(len_src.view(-1, 1, 1, 1)))
                align_prob = (2*attention_probs_src*attention_probs_tgt)/(attention_probs_src+attention_probs_tgt+1e-9)
                return align_matrix, align_prob

            so_loss_src = torch.sum(torch.sum (attention_probs_src*guide, -1), -1).view(-1)
            so_loss_tgt = torch.sum(torch.sum (attention_probs_tgt*guide, -1), -1).view(-1)

            so_loss = so_loss_src/len_src + so_loss_tgt/len_tgt
            so_loss = -torch.mean(so_loss)
            return so_loss






class BertForSO(PreTrainedModel):
    def __init__(self, args, config, model_adapter):
        super().__init__(config)
        self.model = model_adapter
        self.guide_layer = ModelGuideHead()

    def forward(
            self,
            inputs_src,
            inputs_tgt=None,
            labels_src=None,
            labels_tgt=None,
            attention_mask_src=None,
            attention_mask_tgt=None,
            align_layer=6,
            guide=None,
            extraction='softmax', alignment_threshold=0.1,
            entropy_regularization = 0.1,
            marginal_regularization = 0.5,
            mass_transported = 1,
            fertility_distribution = 'l2_norm',
            cost_function = 'cosine_sim',
            position_ids1=None,
            position_ids2=None,
            do_infer=False,
    ):

        loss_fct =CrossEntropyLoss(reduction='none')
        batch_size = inputs_src.size(0)

        output_src = self.model(
            inputs_src,
            attention_mask=attention_mask_src,
            position_ids=position_ids1,
        )

        
        output_tgt = self.model(
            inputs_tgt,
            attention_mask=attention_mask_tgt,
            position_ids=position_ids2,
        )
        if do_infer:
            return output_src, output_tgt
        
        if guide is None:
            raise ValueError('must specify labels for the self-trianing objective')

        

        hidden_states_src = output_src.hidden_states[align_layer]
        hidden_states_tgt = output_tgt.hidden_states[align_layer]


        sco_loss = self.guide_layer(hidden_states_src, hidden_states_tgt, inputs_src, inputs_tgt, guide=guide,
                                    extraction=extraction, alignment_threshold=alignment_threshold,                                                    entropy_regularization = entropy_regularization,
                                    fertility_distribution = fertility_distribution,
                                    marginal_regularization = marginal_regularization,
                                    mass_transported = 1,
                                    cost_function = cost_function)
        return sco_loss

    def save_adapter(self, save_directory, adapter_name):
        self.model.save_adapter(save_directory, adapter_name)
        
    def get_aligned_word(self, inputs_src, inputs_tgt, bpe2word_map_src, bpe2word_map_tgt, device, src_len, tgt_len,
                         align_layer=6, extraction='softmax', alignment_threshold=0.1, 
                         entropy_regularization = 0.1,
                         marginal_regularization = 0.5,
                         mass_transported = 1,                         
                         fertility_distribution = 'l2_norm',
                         cost_function = 'cosine_sim',
                         test=False, output_prob=False,
                         word_aligns=None, pairs_len=None):
        batch_size = inputs_src.size(0)
        bpelen_src, bpelen_tgt = inputs_src.size(1) - 2, inputs_tgt.size(1) - 2
        if word_aligns is None:
            inputs_src = inputs_src.to(dtype=torch.long, device=device).clone()
            inputs_tgt = inputs_tgt.to(dtype=torch.long, device=device).clone()

            with torch.no_grad():
                outputs_src = self.model(
                    inputs_src,
                    attention_mask=(inputs_src != PAD_ID),
                )
                outputs_tgt = self.model(
                    inputs_tgt,
                    attention_mask=(inputs_tgt != PAD_ID),
                )


                hidden_states_src = outputs_src.hidden_states[align_layer]
                hidden_states_tgt = outputs_tgt.hidden_states[align_layer]

                attention_probs_inter = self.guide_layer(hidden_states_src, hidden_states_tgt, inputs_src, inputs_tgt,
                                                         extraction=extraction, alignment_threshold=alignment_threshold,
                                                         entropy_regularization = entropy_regularization,
                                                         marginal_regularization = marginal_regularization,
                                                         mass_transported = 1,
                                                         output_prob=output_prob)
                if output_prob:
                    attention_probs_inter, alignment_probs = attention_probs_inter
                    alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]
                attention_probs_inter = attention_probs_inter.float()

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

        if test:
            
            return word_aligns



        guide = torch.zeros(batch_size, 1, src_len, tgt_len)
        for idx, (word_align, b2w_src, b2w_tgt) in enumerate(zip(word_aligns, bpe2word_map_src, bpe2word_map_tgt)):
            len_src = min(bpelen_src, len(b2w_src))
            len_tgt = min(bpelen_tgt, len(b2w_tgt))

            for i in range(len_src):
                for j in range(len_tgt):
                    if (b2w_src[i], b2w_tgt[j]) in word_align:
                        guide[idx, 0, i + 1, j + 1] = 1.0



        return guide

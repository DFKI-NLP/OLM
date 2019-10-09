from typing import List

from collections import namedtuple
import torch
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer, BertForMaskedLM

Candidate = namedtuple("Candidate", ["tokens", "id", "replaced_index", "weight"])


def get_candidates(tokens: List[str], input_id: int, bert: BertForMaskedLM,
                   tokenizer: BertTokenizer, n_samples: int, replace_subwords: bool,
                   cuda_device: int, unknown: bool = False,
                   verbose: bool = False) -> List[Candidate]:
    vocab_size = len(tokenizer.vocab)

    candidates = []
    # add original sentence to candidates (with replacement index == -1)
    candidates.append(Candidate(tokens=tokens,
                                id=input_id,
                                replaced_index=-1,
                                weight=1.))

    # for t, _ in enumerate(subword_tokens):
    for t, _ in enumerate(tokens):
        tokens_with_mask = list(tokens)

        if unknown:
            tokens_with_mask[t] = tokenizer.unk_token
            candidate = Candidate(tokens=tokens_with_mask,
                                  id=input_id,
                                  replaced_index=t,
                                  weight=n_samples)
            candidates.append(candidate)
            continue

        tokens_with_mask[t] = tokenizer.mask_token

        subword_tokens = tokenizer.tokenize(" ".join(tokens_with_mask))
        masked_index = subword_tokens.index(tokenizer.mask_token)

        if verbose:
            print(f"Subword tokens: {subword_tokens}")
            print(f"Masked index: {masked_index}")

        p_with_masked_token = prob_for_index_at_layer(subword_tokens,
                                                      index=masked_index,
                                                      mask_index=True,
                                                      bert=bert,
                                                      tokenizer=tokenizer,
                                                      cuda_device=cuda_device,
                                                      verbose=verbose)

        # TODO: replace by something smarter (return Candidate [tokens, replaced_index, weight])
        # TODO: also make sure that filter subword replacement (seq len original != seq len candidate)
        # sample n_samples times from pt
        # predict and get probability of class c at token_index
        sampled_token_indices = np.random.choice(vocab_size, size=n_samples, p=p_with_masked_token)
        unique_token_indices, unique_tokens_counts = np.unique(sampled_token_indices, return_counts=True)
        for unique_token_index, unique_token_count in zip(unique_token_indices, unique_tokens_counts):
            sampled_token = tokenizer.convert_ids_to_tokens([unique_token_index])[0]

            # assert not sampled_token.startswith("##")

            tokens_with_replacement = list(tokens)
            tokens_with_replacement[t] = sampled_token
            candidate = Candidate(tokens=tokens_with_replacement,
                                  id=input_id,
                                  replaced_index=t,
                                  weight=unique_token_count)
            candidates.append(candidate)

    if verbose:
        print(candidates)

    return candidates


def prob_for_index_at_layer(tokens: List[str], index: int, mask_index: bool,
                            bert: BertForMaskedLM, tokenizer: BertTokenizer,
                            cuda_device: int, verbose: bool) -> np.array:

    if mask_index:
        tokens = list(tokens)
        tokens[index] = tokenizer.mask_token

    input_token_ids = tokenizer.encode(text=tokens,
                                       add_special_tokens=True,
                                       return_tensors="pt").to(cuda_device)

    if verbose:
        print(f"Input tokens: {tokens}")
        print(f"Input token ids: {input_token_ids}")

    with torch.no_grad():
        logits = bert(input_token_ids)[0]

    probabilities = F.softmax(logits[0, 1:-1], dim=1)[index].detach().cpu().numpy()

    return probabilities

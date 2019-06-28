from typing import List

from collections import namedtuple
import torch
import numpy as np
import torch.nn.functional as F

from pytorch_pretrained_bert import BertTokenizer
from xbert.modeling import BertForMaskedLMLayer

Candidate = namedtuple("Candidate", ["tokens", "id", "replaced_index", "weight"])


def get_candidates(tokens: List[str], input_id: int, bert: BertForMaskedLMLayer,
                   tokenizer: BertTokenizer, n_samples: int, replace_subwords: bool,
                   cuda_device: int, verbose: bool = False) -> List[Candidate]:
    vocab_size = len(tokenizer.vocab)

    # we have to retokenize the input because BERT uses subword tokens
    # subword_tokens = tokenizer.tokenize(" ".join(tokens))

    # if verbose:
    #     print(f"Subword tokens: {subword_tokens}")

    # assert len(tokens) == len(subword_tokens)

    candidates = []
    # add original sentence to candidates (with replacement index == -1)
    candidates.append(Candidate(tokens=tokens,
                                id=input_id,
                                replaced_index=-1,
                                weight=1.))

    # for t, _ in enumerate(subword_tokens):
    for t, _ in enumerate(tokens):
        tokens_with_mask = list(tokens)
        tokens_with_mask[t] = "[MASK]"

        subword_tokens = tokenizer.tokenize(" ".join(tokens_with_mask))
        masked_index = subword_tokens.index("[MASK]")

        if verbose:
            print(f"Subword tokens: {subword_tokens}")

        p_with_masked_token = prob_for_index_at_layer(subword_tokens,
                                                      index=masked_index,
                                                      layer=-1,
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

    return candidates


def prob_for_index_at_layer(tokens: List[str], index: int, layer: int, mask_index: bool,
                            bert: BertForMaskedLMLayer, tokenizer: BertTokenizer,
                            cuda_device: int, verbose: bool) -> np.array:

    if mask_index:
        tokens = list(tokens)
        tokens[index] = "[MASK]"

    input_tokens = ["[CLS]"] + tokens + ["[SEP]"]

    if verbose:
        print(f"Input tokens: {input_tokens}")

    input_token_indices = torch.tensor([tokenizer.convert_tokens_to_ids(input_tokens)],
                                       device=cuda_device)

    with torch.no_grad():
        logits = bert(input_token_indices, layer=layer)[0]

    probabilities = F.softmax(logits[0, 1:-1], dim=1)[index].detach().cpu().numpy()

    return probabilities

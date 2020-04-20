from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from overrides import overrides

from transformers import BertTokenizer, BertForMaskedLM

from olm import InputInstance, OccludedInstance
from olm.occlusion import Strategy, OcclusionStrategy


@Strategy.register("bert_lm_sampling")
class BertLmSampling(OcclusionStrategy):
    def __init__(self,
                 bert_model: str = "bert-base-uncased",
                 cuda_device: int = -1,
                 n_samples: int = 100,
                 verbose: bool = False,
                 std: bool = False,
                 scoring_method=lambda x: x) -> None:
        super().__init__(n_samples, std, scoring_method)
        bert = BertForMaskedLM.from_pretrained(bert_model)
        bert.eval()
        self.bert = bert.to(cuda_device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.cuda_device = cuda_device
        self.n_samples = n_samples
        self.verbose = verbose
        self.vocab_size = len(self.tokenizer.vocab)

    @overrides
    def get_candidate_instances(self,
                                input_instance: InputInstance) -> List[OccludedInstance]:
        # add original sentence to candidates
        occluded_instances = [OccludedInstance.from_input_instance(input_instance)]

        for field_name, token_field in input_instance.token_fields.items():
            for token_idx in range(len(token_field.tokens)):
                tokens_with_mask = list(token_field.tokens)
                tokens_with_mask[token_idx] = self.tokenizer.mask_token

                subword_tokens = self.tokenizer.tokenize(" ".join(tokens_with_mask))
                masked_index = subword_tokens.index(self.tokenizer.mask_token)

                if self.verbose:
                    print(f"Subword tokens: {subword_tokens}")
                    print(f"Masked index: {masked_index}")

                p_with_masked_token = self.prob_for_index(subword_tokens,
                                                          index=masked_index,
                                                          mask_index=True)

                # TODO: also make sure that filter subword replacement (seq len original != seq len candidate)
                # sample n_samples times from pt
                # predict and get probability of class c at token_index
                sampled_token_indices = np.random.choice(self.vocab_size, size=self.n_samples, p=p_with_masked_token)
                unique_token_indices, unique_tokens_counts = np.unique(sampled_token_indices, return_counts=True)
                for unique_token_index, unique_token_count in zip(unique_token_indices, unique_tokens_counts):
                    sampled_token = self.tokenizer.convert_ids_to_tokens([unique_token_index])[0]

                    # assert not sampled_token.startswith("##")

                    tokens_with_replacement = list(token_field.tokens)
                    tokens_with_replacement[token_idx] = sampled_token
                    occluded_inst = OccludedInstance.from_input_instance(
                            input_instance,
                            occlude_token=sampled_token,
                            occlude_field_index=(field_name, token_idx),
                            weight=float(unique_token_count))
                    occluded_instances.append(occluded_inst)

        if self.verbose:
            print(occluded_instances)

        return occluded_instances

    def prob_for_index(self,
                       tokens: List[str],
                       index: int,
                       mask_index: bool) -> np.array:

        if mask_index:
            tokens = list(tokens)
            tokens[index] = self.tokenizer.mask_token

        input_token_ids = self.tokenizer.encode(text=tokens,
                                                add_special_tokens=True,
                                                return_tensors="pt").to(self.cuda_device)

        if self.verbose:
            print(f"Input tokens: {tokens}")
            print(f"Input token ids: {input_token_ids}")

        with torch.no_grad():
            logits = self.bert(input_token_ids)[0]

        probabilities = F.softmax(logits[0, 1:-1], dim=1)[index].detach().cpu().numpy()

        return probabilities

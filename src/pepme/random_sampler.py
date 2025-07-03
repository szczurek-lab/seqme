from __future__ import annotations

from collections import Counter

import torch
from torch.distributions.categorical import Categorical


class RandomAASeqSampler:
    def __init__(
        self,
        lengths: dict[int, int] | dict[int, float],
        aa_frequencies: dict[str, int] | dict[str, float],
    ):
        lengths_normalizing_factor: float = sum(float(x) for x in lengths.values())
        aa_frequencies_normalizing_denominator: float = sum(
            float(x) for x in aa_frequencies.values()
        )

        self.lengths_probs_tensor = torch.zeros(max(lengths.keys()) + 1)
        for len_k, len_v in lengths.items():
            self.lengths_probs_tensor[len_k] = float(len_v) / lengths_normalizing_factor

        self.aa_to_id_map: dict[str, int] = {
            aa: id for id, aa in enumerate(aa_frequencies.keys())
        }
        self.id_to_aa_map: dict[int, str] = {v: k for k, v in self.aa_to_id_map.items()}

        self.aa_frequencies_probs_tensor = torch.zeros(len(self.aa_to_id_map))
        for freq_k, freq_v in aa_frequencies.items():
            self.aa_frequencies_probs_tensor[self.aa_to_id_map[freq_k]] = (
                float(freq_v) / aa_frequencies_normalizing_denominator
            )

    @staticmethod
    def from_reference(reference: list[str]) -> RandomAASeqSampler:
        lengths: Counter[int] = Counter(len(x) for x in reference)
        aa_frequencies: Counter[str] = Counter()
        for x in reference:
            aa_frequencies.update(Counter(x))
        return RandomAASeqSampler(lengths=lengths, aa_frequencies=aa_frequencies)

    def sample(self, n: int) -> list[str]:
        lengths_sampler = Categorical(self.lengths_probs_tensor)
        aa_ids_sampler = Categorical(self.aa_frequencies_probs_tensor)
        lengths = lengths_sampler.sample((n, 1))
        aa_ids = aa_ids_sampler.sample((lengths.sum(),))
        it = 0
        ret = []
        for seq_len in lengths:
            ids_seq = aa_ids[it : it + seq_len].tolist()
            aa_seq = "".join(self.id_to_aa_map[id] for id in ids_seq)
            ret.append(aa_seq)
            it += seq_len
        return ret

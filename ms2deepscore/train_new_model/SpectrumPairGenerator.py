import json
from collections import Counter
from typing import List, Tuple, Union
from collections import defaultdict
import numpy as np
from matchms import Spectrum
from ms2deepscore.train_new_model.pair_data_persistence import SelectedPairSchedule


class SpectrumPairGenerator:
    def __init__(self, selected_inchikey_pairs: Union[List[Tuple[str, str, float]], SelectedPairSchedule], spectra,
                 shuffle: bool = True, random_seed: int = 0):
        """
        Parameters
        ----------
        selected_inchikey_pairs:
            A list with tuples encoding inchikey pairs like: (inchikey1, inchikey2, tanimoto_score)
        """
        self.selected_inchikey_pairs = selected_inchikey_pairs
        self.spectra = spectra
        self.spectrum_inchikeys = np.array([s.get("inchikey")[:14] for s in self.spectra])
        self.shuffle = shuffle
        self.random_nr_generator = np.random.default_rng(random_seed)
        self._idx = 0
        self._compact_schedule = isinstance(self.selected_inchikey_pairs, SelectedPairSchedule)
        if self._compact_schedule:
            self._pair_order = np.arange(len(self.selected_inchikey_pairs), dtype=np.int64)
            if self.shuffle:
                self.random_nr_generator.shuffle(self._pair_order)
        elif self.shuffle:
            # Preserve the historical behaviour for the legacy list representation.
            self.random_nr_generator.shuffle(self.selected_inchikey_pairs)
        
        # create a mapping from inchikey to spectrum indices for efficient retrieval
        self.spectra_by_inchikey = defaultdict(list)
        for i, spectrum in enumerate(self.spectra):
            inchikey = spectrum.get("inchikey")
            if inchikey is not None:
                self.spectra_by_inchikey[inchikey[:14]].append(i)

    def __iter__(self):
        return self

    def __next__(self):
        # reshuffle when we've gone through everything
        if self._idx >= len(self.selected_inchikey_pairs):
            self._idx = 0
            if self.shuffle:
                if self._compact_schedule:
                    self.random_nr_generator.shuffle(self._pair_order)
                else:
                    self.random_nr_generator.shuffle(self.selected_inchikey_pairs)

        pair_index = int(self._pair_order[self._idx]) if self._compact_schedule else self._idx
        inchikey1, inchikey2, tanimoto_score = self.selected_inchikey_pairs[pair_index]
        spectrum1 = self._get_spectrum_with_inchikey(inchikey1, self.random_nr_generator)
        spectrum2 = self._get_spectrum_with_inchikey(inchikey2, self.random_nr_generator)
        self._idx += 1
        return spectrum1, spectrum2, tanimoto_score

    def __len__(self):
        return len(self.selected_inchikey_pairs)

    def __str__(self):
        return f"SpectrumPairGenerator with {len(self.selected_inchikey_pairs)} pairs available"

    def get_scores(self):
        if self._compact_schedule:
            return np.asarray(self.selected_inchikey_pairs.scores).tolist()
        return [score for _, _, score in self.selected_inchikey_pairs]

    def get_inchikey_counts(self) -> Counter:
        """returns the frequency each inchikey occurs"""
        inchikeys = Counter()
        for inchikey_1, inchikey_2, _ in self.selected_inchikey_pairs:
            inchikeys[inchikey_1] += 1
            inchikeys[inchikey_2] += 1
        return inchikeys

    def get_scores_per_inchikey(self):
        inchikey_scores = {}
        for inchikey_1, inchikey_2, score in self.selected_inchikey_pairs:
            if inchikey_1 in inchikey_scores:
                inchikey_scores[inchikey_1].append(score)
            else:
                inchikey_scores[inchikey_1] = [score]
            if inchikey_2 in inchikey_scores:
                inchikey_scores[inchikey_2].append(score)
            else:
                inchikey_scores[inchikey_2] = [score]
        return inchikey_scores

    def save_as_json(self, file_name):
        data_for_json = [(item[0], item[1], float(item[2])) for item in self.selected_inchikey_pairs]

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data_for_json, f)

    def _get_spectrum_with_inchikey(self, inchikey: str, random_number_generator) -> Spectrum:
        """
        Get a random spectrum matching the `inchikey` argument.

        NB: A compound (identified by an
        inchikey) can have multiple measured spectrums in a binned spectrum dataset.
        """
        indices = self.spectra_by_inchikey.get(inchikey)
        if not indices:
            raise ValueError(f"No matching inchikey found: {inchikey}")
        return self.spectra[random_number_generator.choice(indices)]

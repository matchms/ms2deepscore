import json
from collections import Counter
from typing import List, Tuple


class InchikeyPairGenerator:
    def __init__(self, selected_inchikey_pairs: List[Tuple[str, str, float]]):
        """
        Parameters
        ----------
        selected_inchikey_pairs:
            A list with tuples encoding inchikey pairs like: (inchikey1, inchikey2, tanimoto_score)
        """
        self.selected_inchikey_pairs = selected_inchikey_pairs

    def generator(self, shuffle: bool, random_nr_generator):
        """Infinite generator to loop through all inchikeys.
        After looping through all inchikeys the order is shuffled.
        """
        while True:
            if shuffle:
                random_nr_generator.shuffle(self.selected_inchikey_pairs)

            for inchikey1, inchikey2, tanimoto_score in self.selected_inchikey_pairs:
                yield inchikey1, inchikey2, tanimoto_score

    def __len__(self):
        return len(self.selected_inchikey_pairs)

    def __str__(self):
        return f"InchikeyPairGenerator with {len(self.selected_inchikey_pairs)} pairs available"

    def get_scores(self):
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
                inchikey_scores[inchikey_1] = []
            if inchikey_2 in inchikey_scores:
                inchikey_scores[inchikey_2].append(score)
            else:
                inchikey_scores[inchikey_2] = []
        return inchikey_scores

    def save_as_json(self, file_name):
        data_for_json = [(item[0], item[1], float(item[2])) for item in self.selected_inchikey_pairs]

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data_for_json, f)

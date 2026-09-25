import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Seeded pseudo-random words over a tiny alphabet: overlapping pairs across
# many words plus lots of equal counts, so merge order hinges on tie-breaking.
_SNIPPET = """
import random
from pipeline.morphemes import train_bpe
rng = random.Random(0)
text = " ".join("".join(rng.choice("abcdef") for _ in range(rng.randint(2, 7))) for _ in range(600))
print(train_bpe(text))
"""


def _merges_under_hash_seed(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    return subprocess.run([sys.executable, "-c", _SNIPPET], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout


class TrainBpeDeterminismTest(unittest.TestCase):
    def test_merges_independent_of_hash_seed(self):
        # Morpheme features are computed at both training and inference time;
        # hash-order-dependent merges made the same text featurize differently
        # in every process.
        results = {_merges_under_hash_seed(seed) for seed in range(6)}
        self.assertEqual(len(results), 1, results)


if __name__ == "__main__":
    unittest.main()

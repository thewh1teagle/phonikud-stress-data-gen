"""
wget https://huggingface.co/datasets/thewh1teagle/phonikud-data/resolve/main/knesset_nikud_v6.txt.7z
7z x knesset_nikud_v6.txt.7z
"""

from collections import defaultdict
import re

from phonikud import lexicon
from tqdm import tqdm

import config

def main():
    data = defaultdict(int)
    with open("knesset_nikud_v6.txt", "r") as f:
        total_lines = sum(1 for _ in f) # type: ignore
        f.seek(0)
        for line in tqdm(f, total=total_lines):
            words = re.findall(config.HE_PATTERN, line)
            
            without_prefix = []
            for word in words:
                # remove prefix diacritic
                prefix_pos = word.find(lexicon.PREFIX_DIACRITIC)
                if prefix_pos != -1:
                    # remove word prefix letters and diacritics
                    word = word[prefix_pos + 1 :]
                if lexicon.HATAMA_DIACRITIC in word:
                    # Only include words with hatama diacritic
                    without_prefix.append(word)
            for word in without_prefix:
                data[word] += 1
    # sort by count
    data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    with open("top_stressed.txt", "w") as f:
        for word, count in data:
            f.write(f"{word}\t{count}\n")


if __name__ == "__main__":
    main()

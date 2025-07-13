"""
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
"""
from phonikud_onnx import Phonikud
import json
import config
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input json file from create_stressed_sentences.py")
    parser.add_argument("output", type=str, help="output text file") # will create lines with sentences with diacritics
    args = parser.parse_args()
    
    phonikud = Phonikud("./phonikud-1.0.int8.onnx")
    with open(args.input, "r") as f:
        data: list[dict] = json.load(f)

    
    with open(args.output, "w") as f:
        for item in tqdm(data):
            # filter out sentences that don't contain any of the allowed characters
            sentences = [sentence for sentence in item["sentences"] if any(char in sentence for char in config.HEBREW_LETTERS)]
            if not sentences:
                continue
            for sentence in sentences:
                with_diacritics = phonikud.add_diacritics(sentence)
                f.write(with_diacritics + "\n")
                f.flush()

if __name__ == "__main__":
    main()

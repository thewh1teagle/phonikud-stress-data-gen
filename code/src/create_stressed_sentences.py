"""
https://platform.openai.com/settings/organization/api-keys
uv run code/src/create_stressed_sentences.py ./data/200_stressed.txt output.json
"""
from openai.types.chat import ChatCompletion
from phonikud import lexicon
import os
import dotenv
import config
import argparse
import re
from openai import OpenAI
from pydantic import BaseModel
import json

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    


class SentenceGenerateResult(BaseModel):
    sentences: list[str]

def ask_openai(prompt: str, model: str = config.DEFAULT_MODEL):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response: ChatCompletion = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=SentenceGenerateResult,
    )
    input_tokens = response.usage.prompt_tokens # type: ignore
    output_tokens = response.usage.completion_tokens # type: ignore

    return {
        "content": response.choices[0].message.parsed.sentences, # type: ignore
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

def get_input_words(input_file: str):
    words = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, _ = line.split('\t') # word, count
            word = re.sub(lexicon.HE_NIKUD_PATTERN, "", word)
            word = word.strip()
            if not word:
                continue
            words.append(word)
    return words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    
    words = get_input_words(args.input)
    
    with open(args.output, "w") as f:
        
        data = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for word in words:  
            try:
                prompt = config.OPENAI_PROMPT_TEMPLATE.format(word=word)
                response = ask_openai(prompt)
                data.append({
                    "word": word,
                    "sentences": response["content"],
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                })
                total_input_tokens += response["input_tokens"]
                total_output_tokens += response["output_tokens"]
                total_input_cost = total_input_tokens * config.MODELS[config.DEFAULT_MODEL]['cost']['input']
                total_output_cost = total_output_tokens * config.MODELS[config.DEFAULT_MODEL]['cost']['output']
                total_cost = total_input_cost + total_output_cost
                print(f"Total cost: {total_cost:.6f} USD")
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()

            except Exception as e:
                print(f"Error: {e}")
                print(f"Word: {word}")
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")

    

if __name__ == "__main__":
    main()
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from pathlib import Path
import sys
import time

def get_device(cuda_index: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_index}")
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")

def preprocess_text(text: str) -> str:
    # Reemplaza URLs por 'http'
    text = str(text)
    if not text:
        return ""
    tokens = ['http' if t.startswith('http') else t for t in text.split()]
    return " ".join(tokens)

def index_word(text: str, word: str) -> str:
    index = text.find(word)
    if index == -1:
        return text
    return text[index + len(word):]

def load_model_and_tokenizer(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 256
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return tokenizer, config, model

def get_scores(text: str, tokenizer, model, device):
    #text = preprocess_text(text)
    encoded_input = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output.logits[0].cpu().numpy()
    return softmax(scores)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <cuda_index>")
        sys.exit(1)

    cuda_index = int(sys.argv[1])
    device = get_device(cuda_index)

    tasks = ["sentiment", "emotion"]
    for task in tasks:
        if task == "sentiment":
            model_name = f"cardiffnlp/twitter-xlm-roberta-base-{task}"
        elif task == "emotion":
            model_name = f"cardiffnlp/twitter-roberta-base-{task}"
        else:
            continue

        print(f"Loading model and tokenizer for task: {task}")
        tokenizer, config, model = load_model_and_tokenizer(model_name, device)
        print("Labels:", config.id2label)

        # Define experiment variables
        prompt = "f3"
        template = "EEC"
        name_experiment = "few_vs_zero"

        base_path = Path(f"/home/tquiroga/llm_test_1/Adaptation/adaptation_llm/Pertubation/{template}/{prompt}/{name_experiment}")
        csv_files = list(base_path.glob("*.csv"))

        # Construir diccionario de perturbaciones filtradas
        def filter_key(k):
            return "mixtral" in k or "few" in k

        dict_perturbation = {
            index_word(f.stem, "pertubation_"): str(f)
            for f in csv_files
            if filter_key(f.stem)
        }

        print(f"Perturbations found: {list(dict_perturbation.keys())}")

        output_dir = Path(f"/home/tquiroga/llm_test_1/Adaptation/Scores/{model_name.replace('/', '-')}/{template}/{prompt}/{name_experiment}")
        output_dir.mkdir(parents=True, exist_ok=True)

        time_log_file = Path(f"times_llm_{cuda_index}_country_few.txt")
        time_log_file.write_text("")  # Clear file at start

        for i, (domain, path_csv) in enumerate(dict_perturbation.items(), 1):
            print(f"\n[{i}/{len(dict_perturbation)}] Processing domain: {domain}")
            print(f"Model: {model_name}")
            print(f"Path: {path_csv}")

            start_time = time.time()
            df = pd.read_csv(path_csv).reset_index(drop=True)
            print("Unique template types:", df["template_type"].unique().tolist())

            df_scores = {label: [] for label in config.id2label}

            for text in tqdm(df["template"], desc="Scoring templates"):
                scores = get_scores(text, tokenizer, model, device)
                for label_idx, label_name in config.id2label.items():
                    df_scores[label_name].append(scores[label_idx].item())

            df_score = pd.DataFrame(df_scores)
            # Copiar columnas importantes
            for col in ["template", "template_index", "template_type", "identity_term", "group"]:
                df_score[col] = df[col]

            df_score.reset_index(drop=True, inplace=True)
            save_path = output_dir / f"scores_{domain}.csv"
            print(f"Saving scores to: {save_path}")
            df_score.to_csv(save_path, index=False)

            elapsed = time.time() - start_time
            print(f"Elapsed time: {elapsed:.2f} seconds")
            with time_log_file.open("a") as f:
                f.write(f"{domain} {elapsed:.2f}\n")

if __name__ == "__main__":
    main()
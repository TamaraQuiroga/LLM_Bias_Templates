import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time
import sys

# Part of this code was based on https://github.com/ascamara/ml-intersectionality
# Seed para reproducibilidad
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def valor_domain(file_path: str) -> str:
    if "pertubation_EEC" in file_path:
        return "EEC"
    elif "wikipedia_talks" in file_path:
        return "wikipedia_talks"
    else:
        return "tweets"

def get_hidden_size(model, tokenizer):
    """Obtiene el tamaño de la última capa oculta del modelo."""
    with torch.no_grad():
        sample = tokenizer('Sample sentence for tokenizer', padding='max_length', max_length=64, truncation=True, return_tensors='pt')
        outputs = model(**sample, output_hidden_states=True)
        hidden_size = outputs.hidden_states[-1][:, 0, :].shape[-1]
    return hidden_size

class ModelRegressor(torch.nn.Module):
    def __init__(self, model_class, tokenizer, pretrained_weights: str, freeze: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = model_class.from_pretrained(pretrained_weights)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = get_hidden_size(self.bert, self.tokenizer)
        self.fc0 = torch.nn.Linear(hidden_size, 128)
        self.activation = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(128, 1)

    def forward(self, texts):
        # Tokenizar sin gradiente
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(device)

        outputs = self.bert(**inputs, output_hidden_states=True)
        pooled_output = outputs.hidden_states[-1][:, 0, :]
        x = self.activation(self.fc0(pooled_output))
        x = self.fc1(x)
        return x.squeeze(-1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <cuda_index>")
        sys.exit(1)

    global device
    cuda_index = int(sys.argv[1])
    device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    load = True
    dict_path_load = {
        "bert-base-cased": "/home/tquiroga/bert_fair/model_trained/model_valence_bert-base-cased_10.pth",
        "distilbert/distilbert-base-cased": "/home/tquiroga/bert_fair/model_trained/model_valence_distilbert-base-cased_10.pth",
        "google-bert/bert-base-multilingual-cased": "/home/tquiroga/bert_fair/model_trained/model_valence_multibert-base-cased_10.pth",
        "cardiffnlp/twitter-xlm-roberta-base": "/home/tquiroga/bert_fair/model_trained/model_valence_cardiffnlp_twitter-xlm-roberta-base_10.pth",
    }

    template_ = "EEC"
    prompt_ = "f3"
    name_experiment = "IMBD"

    base_path = Path(f"/home/tquiroga/llm_test_1/Adaptation/adaptation_llm/Pertubation/{template_}/{prompt_}/{name_experiment}")
    csv_files = list(base_path.glob("*.csv"))
    dict_perturbation = {
        f.name.replace(f"pertubation_", "").replace(".csv", ""): str(f)
        for f in csv_files
    }

    print(f"Perturbations found: {list(dict_perturbation.keys())}")

    for name_model in dict_path_load:
        print(f"\nEvaluating model: {name_model}")
        tokenizer = AutoTokenizer.from_pretrained(name_model)
        model_class = AutoModelForMaskedLM
        model = ModelRegressor(model_class, tokenizer, name_model, freeze=False).to(device)

        if load:
            model.load_state_dict(torch.load(dict_path_load[name_model], map_location=device))

        output_base = Path(f"/home/tquiroga/llm_test_1/Adaptation/Scores/{name_model.replace('/', '-')}/{template_}/{prompt_}/{name_experiment}")
        output_base.mkdir(parents=True, exist_ok=True)

        with open(f"times_ft_llm_{cuda_index}.txt", "a") as time_log:
            for idx, (domain_, path_) in enumerate(dict_perturbation.items(), start=1):
                start_time = time.time()
                print(f"Processing ({idx}/{len(dict_perturbation)}): {domain_}")
                print(f"Input path: {path_}")

                df = pd.read_csv(path_).reset_index(drop=True)
                print("Template types found:", df["template_type"].unique().tolist())

                df_scores = []
                for text in tqdm(df["template"], desc="Predicting scores"):
                    score = model([text]).item()
                    df_scores.append(score)

                df_result = pd.DataFrame({
                    "score": df_scores,
                    "template": df["template"],
                    "template_type": df["template_type"],
                    "identity_term": df["identity_term"],
                    "group": df["group"],
                    "template_index": df["template_index"],
                    "domain": valor_domain(path_),
                })

                save_path = output_base / f"scores_{domain_}.csv"
                df_result.to_csv(save_path, index=False)
                print(f"Saved results to {save_path}")

                elapsed = time.time() - start_time
                print(f"Time elapsed: {elapsed:.2f} seconds\n")
                time_log.write(f"{domain_}, {name_model.replace('/', '-')}, {elapsed:.2f}\n")

if __name__ == "__main__":
    main()
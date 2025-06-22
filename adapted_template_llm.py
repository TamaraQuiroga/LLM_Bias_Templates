import subprocess
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import json
import gc
import time
from pathlib import Path
from tqdm import tqdm
from utils import clean_response, check_different, check_name, format_few_example

# Login to HuggingFace CLI
command = [
    "huggingface-cli", 
    "login", 
    "--token", 
    "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", 
    "--add-to-git-credential"
]

result = subprocess.run(command, capture_output=True, text=True)
if result.returncode == 0:
    print("Login successful:")
    print(result.stdout)
else:
    print("Login error:")
    print(result.stderr)


class AdaptedTemplateLLM:
    def __init__(self,
                 name_template,
                 name_domain,
                 model_name,
                 prompt_name,
                 path_template,
                 path_domain_examples,
                 path_few_examples=None,
                 path_names="Files/name_male_popular.csv",
                 experiment_name="",
                 few_examples=False):

        # Experiment configuration
        self.name_template = name_template  # e.g., "EEC" or "IPTTS"
        self.name_domain = name_domain      # e.g., "tweets", "wikipedia_talks"
        self.model_name = model_name        # e.g., "llama3_8", "mixtral"
        self.prompt_name = prompt_name      # e.g., "f3"
        self.experiment_name = experiment_name
        self.few_examples = few_examples

        # File paths
        self.path_template = path_template
        self.path_domain_examples = path_domain_examples
        self.path_few_examples = path_few_examples
        self.path_names = path_names

        # DataFrames and LLM components
        self.df_template = None
        self.df_domain_examples = None
        self.text_prompt = None

        self.model = None
        self.tokenizer = None

    def load_model(self, model_key):
        if model_key == "llama3_8":
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
            model_args = {"load_in_8bit": True, "device_map": "cuda:0"}

        elif model_key == "llama3_70":
            model_name = "meta-llama/Llama-3.3-70B-Instruct"
            model_args = {"load_in_8bit": True, "device_map": "auto"}

        elif model_key == "mixtral":
            model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            model_args = {"load_in_4bit": True, "device_map": "auto"}

        else:
            raise ValueError("Unsupported model key. Use 'llama3_8', 'llama3_70', or 'mixtral'.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            pad_token_id=self.tokenizer.eos_token_id,
            **model_args
        )

    def response_model(self, sentence, device, max_tokens):
        inputs = self.tokenizer(sentence, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[-1] + max_tokens + 8,
            do_sample=True,
            temperature=0.01,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def load_domain_examples(self):
        df = pd.read_csv(self.path_domain_examples)
        df = df.rename(columns={"tweet": "examples"}).drop_duplicates()
        self.df_domain_examples = df
        self.df_domain_examples_toxic = df[df["toxicity"] == 1]
        self.df_domain_examples_nontoxic = df[df["toxicity"] == 0]

    def load_template_df(self):
        df = pd.read_csv(self.path_template)[:2]
        if self.name_template != "IPTTS":
            df = df[['sentence', 'person']]
            df["identity_term"] = df["person"]
        else:
            df = df[['sentence', 'person', 'toxicity']]
        self.df_template = df.reset_index(drop=True)

    def load_full_template_df(self, path):
        self.df_template = pd.read_csv(path)

    def load_prompt_text(self):
        with open('Prompts/prompts_text.json') as f:
            prompts = json.load(f)
        self.text_prompt = prompts[self.prompt_name]

    def load_few_shot_examples(self, n=3):
        df = pd.read_csv(self.path_few_examples, sep=";", names=["rewrite", "original"])
        df["few_examples"] = df.apply(lambda x: format_few_example(x["rewrite"], x["original"]), axis=1)
        return [df["few_examples"].sample().iloc[0] for _ in range(n)]

    def replace_names_with_top(self, max_names=200):
        df_names = pd.read_csv(self.path_names)
        top_names = df_names["firstname"].tolist()[:max_names]
        total_templates = self.df_template.shape[0]

        repeat_count = total_templates // max_names
        remainder = total_templates % max_names

        names_list = top_names * repeat_count + top_names[:remainder]
        self.df_template["identity_term"] = names_list
        self.df_template["sentence"] = self.df_template.apply(
            lambda x: x["sentence"].replace(x["person"], x["identity_term"]), axis=1)

    def substitute_examples(self, n, label=None):
        if label is None or label == "nontoxic" or (label == "toxic" and self.df_domain_examples_toxic.empty):
            sample_df = self.df_domain_examples_nontoxic
        else:
            sample_df = self.df_domain_examples_toxic
        return "\n".join(f"{i+1}. {text}" for i, text in enumerate(sample_df.sample(n=n)["examples"].tolist()))

    def load_df_prompt_text(self, model_key, n_examples=15):
        """
        Build the LLM prompt by replacing placeholders in the template with examples and identity terms.
        """
        prompt_template = self.text_prompt
        prompt_col_name = f"prompt_{model_key}_{self.prompt_name}"
        print(f"Generating column: {prompt_col_name}")

        self.df_template[prompt_col_name] = self.df_template["sentence"].apply(
            lambda x: prompt_template.replace("TEMPLATE_ECC", x)
        )
        self.df_template[prompt_col_name] = self.df_template.apply(
            lambda x: x[prompt_col_name].replace("XXXX", x["identity_term"]), axis=1
        )
        self.df_template[prompt_col_name] = self.df_template.apply(
            lambda x: x[prompt_col_name].replace("NNN", str(n_examples)), axis=1
        )

        # Insert domain-specific examples
        if self.name_template == "IPTTS" and not self.df_domain_examples_toxic.empty:
            toxic = [self.substitute_examples(n_examples, label="toxic") for _ in range(len(self.df_template))]
            nontoxic = [self.substitute_examples(n_examples, label="nontoxic") for _ in range(len(self.df_template))]
            self.df_template["examples_prompt"] = [
                toxic[i] if label == "toxic" else nontoxic[i]
                for i, label in enumerate(self.df_template["toxicity"])
            ]
        else:
            self.df_template["examples_prompt"] = [
                self.substitute_examples(n_examples) for _ in range(len(self.df_template))
            ]

        self.df_template[prompt_col_name] = self.df_template.apply(
            lambda x: x[prompt_col_name].replace("YYYY", x["examples_prompt"]), axis=1
        )

        # Adjust for Mixtral model
        if self.model_name == "mixtral":
            self.df_template[prompt_col_name] = self.df_template[prompt_col_name].apply(
                lambda x: f"<s> [INST] {x} [/INST]"
            )

        # Insert few-shot examples if provided
        if self.few_examples:
            few_shot_text = "\n".join(self.load_few_shot_examples())
            self.df_template[prompt_col_name] = self.df_template[prompt_col_name].apply(
                lambda x: x.replace("EEEE", few_shot_text)
            )

        return prompt_col_name

    def generate_template(self, n_examples=15, max_retries=5, prompt_already=None):
        """
        Generate rewritten templates using the loaded LLM and compare them to the originals.
        """

        model_key = self.model_name
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.load_model(model_key)

        # Estimate average output length
        self.df_domain_examples["size"] = self.df_domain_examples["examples"].apply(
            lambda x: self.tokenizer(x, return_tensors="pt").to(device)["input_ids"].shape[-1]
        )
        avg_token_len = int(self.df_domain_examples["size"].mean()) + 2
        print(f"Average token length for generation: {avg_token_len}")

        if prompt_already is None:
            prompt_col = self.load_df_prompt_text(model_key, n_examples)
        else:
            prompt_col = prompt_already

        output_col = "response_" + prompt_col.replace("prompt_", "")
        final_prompt_col = prompt_col + "_final"

        template_texts = self.df_template["sentence"]
        identity_names = self.df_template["identity_term"]
        examples_list = self.df_template["examples_prompt"]
        labels = self.df_template["toxicity"] if "toxicity" in self.df_template.columns else [None] * len(self.df_template)

        responses = []
        final_prompts = []

        output_dir = Path(f"LLM_templates/originales/{self.name_template}/{self.prompt_name}/{self.experiment_name}/")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"PRUEBA_{model_key}_{self.name_domain}.csv"

        # Clear output file
        with open(output_path, "w") as f:
            pass

        for i, prompt in enumerate(tqdm(self.df_template[prompt_col])):
            original_prompt = prompt
            retries = max_retries
            label = labels[i]

            while retries > 0:
                result = self.response_model(prompt, device, avg_token_len)
                cleaned = clean_response(result)

                if check_different(template_texts[i], cleaned) and check_name(identity_names[i], cleaned):
                    break  # Good output
                else:
                    # Replace example block and retry
                    new_examples = self.substitute_examples(n_examples, label)
                    prompt = prompt.replace(examples_list[i], new_examples)
                    examples_list[i] = new_examples
                    retries -= 1

            if retries == 0:
                cleaned = template_texts[i]  # Fallback to original if failed

            responses.append(cleaned)
            final_prompts.append(original_prompt)

            with open(output_path, "a") as f:
                f.write(cleaned + "\n")

        self.df_template[output_col] = responses
        self.df_template[final_prompt_col] = final_prompts

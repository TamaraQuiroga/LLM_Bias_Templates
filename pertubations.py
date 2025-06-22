import pandas as pd
import torch
import json
import re
import spacy
from pathlib import Path
from tqdm import tqdm
from utils import (
    clean_response,
    check_different,
    check_name,
    format_few_example,
    check_name_method,
    clean_arrow
)


class GenderSwapper:
    def __init__(self):
        word_path = "Files/words_zhao.txt"
        df_words = pd.read_csv(word_path, header=None)
        df_words[1] = df_words[1].str.strip()

        self.male_words = self.expand_list(df_words[1])
        self.female_words = self.expand_list(df_words[0])

        self.nlp = spacy.load("en_core_web_lg")

    def expand_list(self, word_list):
        base = [w.strip() for w in word_list]
        return base + [w.capitalize() for w in base] + [w.upper() for w in base]

    def mark_her(self, text):
        modified_text = ""
        for token in self.nlp(text):
            if token.text.lower() == "her":
                tag_marker = "2" if "$" in token.tag_ else "1"
                modified_text += tag_marker + token.text + token.whitespace_
            else:
                modified_text += token.text_with_ws
        return modified_text

    def apply_cda(self, text):
        male_to_female = dict(zip(self.male_words, self.female_words))
        female_to_male = dict(zip(self.female_words, self.male_words))

        # Replace "her" with disambiguated tokens
        her_map = {
            "1her": "him", "2her": "his",
            "1Her": "Him", "2Her": "His",
            "1HER": "HIM", "2HER": "HIS",
            "her": "his"
        }
        text = self.mark_her(text)
        for key, value in her_map.items():
            self.female_words.append(key)
            female_to_male[key] = value

        to_replace = set()
        for word in self.male_words + self.female_words:
            if re.search(rf"\b{word}\b", text):
                to_replace.add(word)

        for word in to_replace:
            if word in male_to_female:
                text = re.sub(rf"\b{word}\b", male_to_female[word], text)
            elif word in female_to_male:
                text = re.sub(rf"\b{word}\b", female_to_male[word], text)

        return text


class TemplatePerturbator:
    def __init__(self, template_path, json_groups_path, response_col="c_response"):
        with open(json_groups_path, 'r') as file:
            self.group_data = json.load(file)

        self.response_col = response_col
        self.df_original = pd.read_csv(template_path)
        self.df_gender_swapped = self.df_original.copy()
        self.df_perturbed = None

    def replace_name(self, name, new_name, text):
        if name.lower() == new_name.lower() and text.startswith(name):
            new_name = new_name.capitalize()

        for variant in [name, name.upper(), name.lower(), name.capitalize()]:
            if variant in text:
                text = text.replace(variant, new_name)
                break
        return text

    def create_perturbed_template(self, df, new_term):
        for col in df.columns:
            if col.startswith(self.response_col):
                break

        result = {
            "template": df.apply(lambda x: self.replace_name(x["identity_term"], new_term, x[col]), axis=1),
            "template_type": col.replace("c_response_", ""),
            "template_index": df["index"]
        }
        return pd.DataFrame(result)

    def standar_names_cols(self):
        if "identity_term" not in self.df_original.columns and "new_name" in self.df_original.columns:
            self.df_original = self.df_original.rename(columns={"new_name": "identity_term"})
        if "identity_term" not in self.df_original.columns and "person" in self.df_original.columns:
            self.df_original = self.df_original.rename(columns={"person": "identity_term"})
        if "template" not in self.df_original.columns and "sentence" in self.df_original.columns:
            self.df_original = self.df_original.rename(columns={"sentence": "template"})

    def generate_all_perturbations(self):
        self.standar_names_cols()
        results = []
        gender_swapper = GenderSwapper()

        for group, values in self.group_data.items():
            terms = values["male"]
            df_source = self.df_gender_swapped if group.startswith("F-") or group == "Female" else self.df_original

            if group in ["Female", "F-Neutral"]:
                df_source[self.response_col] = df_source[self.response_col].apply(gender_swapper.apply_cda)

            for term in terms:
                df_term = self.create_perturbed_template(df_source, term)
                df_term["group"] = group
                df_term["identity_term"] = term
                results.append(df_term)

        df_result = pd.concat(results).reset_index(drop=True)
        df_result["valid"] = df_result.apply(lambda row: check_name_method(row["identity_term"], row["template"]), axis=1)
        assert df_result["valid"].sum() == df_result.shape[0]
        df_result = df_result.drop(columns=["valid"])
        self.df_perturbed = df_result
        return df_result


class PerturbationExperiment:
    def __init__(self, template, prompt, model, domain, experiment, group_json_path, previous_experiment=None):
        self.template = template
        self.prompt = prompt
        self.model = model
        self.domain = domain
        self.experiment = experiment
        self.previous_experiment = previous_experiment or experiment
        self.group_json_path = group_json_path

    def run(self):
        original_path = f"LLM_templates/originales/{self.template}/{self.prompt}/{self.previous_experiment}/llm_template_{self.model}_{self.domain}.csv"
        df = pd.read_csv(original_path)

        if "identity_term" not in df.columns and "new_name" in df.columns:
            df = df.rename(columns={"new_name": "identity_term"})
            df.to_csv(original_path, index=False)

        response_col = f"c_response_{self.model}_{self.prompt}"
        if response_col not in df.columns:
            df[response_col] = df[f"response_{self.model}_{self.prompt}"].str.replace("1.", "", regex=False)

        df[response_col] = df[response_col].apply(clean_arrow)
        df[response_col] = df[response_col].str.replace("{", "").str.replace("}", "").str.replace("1 .", "")

        if "index" not in df.columns:
            df = df.reset_index()
        df.to_csv(original_path, index=False)

        df_eval = df[["sentence", "identity_term", response_col]].copy().reset_index()
        df_eval["check_name"] = df_eval.apply(lambda x: check_name(x["identity_term"], x[response_col]), axis=1)
        df_eval["different"] = df_eval.apply(lambda x: check_different(x["sentence"], x[response_col]), axis=1)

        assert df_eval["check_name"].sum() == df_eval.shape[0]
        print(f"Name check passed: {df_eval['check_name'].sum()}/{df_eval.shape[0]}, Different: {df_eval['different'].sum()}")

        perturber = TemplatePerturbator(original_path, self.group_json_path, response_col=response_col)
        df_perturbed = perturber.generate_all_perturbations()

        output_path = f"LLM_templates/Perturbation/{self.template}/{self.prompt}/{self.experiment}/perturbation_{self.model}_{self.domain}.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df_perturbed.to_csv(output_path, index=False)
        print(f"Perturbation complete. Saved to {output_path}")

    def run_original_template(self):
        path_original_EEC =f"Templates/{self.template}_counterfactual.csv"
        path_pertubation = f"LLM_templates/Perturbation/{self.template}/{self.prompt}/{self.experiment}/perturbation_{self.template}.csv"

        if Path(path_pertubation).exists():
            print(f"Perturbation {self.template} exist.")
        else:
            perturber = TemplatePerturbator(path_original_EEC,self.group_json_path,response_col="template")
            df_perturbed = perturber.generate_all_perturbations()
            df_perturbed.to_csv(path_pertubation, index=False)
            print(f"Perturbation complete. Saved to {path_pertubation}")

    def run_NOEs_template(self):
        path_original_NOEs =f"NOEs/{self.domain}_counterfactual.csv"
        path_pertubation = (f"LLM_templates/Perturbation/{self.template}/{self.prompt}/{self.experiment}/perturbation_{self.domain}.csv")

        if Path(path_pertubation).exists():
            print(f"Perturbation {self.domain} exist.")
        else:
            perturber = TemplatePerturbator(path_original_NOEs,self.group_json_path,response_col="template")
            df_perturbed = perturber.generate_all_perturbations()
            df_perturbed.to_csv(path_pertubation, index=False)
            print(f"Perturbation complete. Saved to {path_pertubation}")

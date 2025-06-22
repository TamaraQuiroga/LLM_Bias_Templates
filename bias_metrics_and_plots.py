import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from utils import clean_unnamed, MAE, MSE



domain_dict = {
    "tweets": "tweets",
    "wikipedia_talks": "wikipedia_talks_nontoxic",
    "IMBD": "IMBD"
}

model_name_map = {
    "bert-base-cased": "BERT",
    "distilbert-distilbert-base-cased": "distil-BERT",
    "cardiffnlp-twitter-xlm-roberta-base": "cardiffnlp-roberta-xlm",
    "cardiffnlp-sentiment": "cardiffnlp-sentiment",
    "cardiffnlp-emotion": "cardiffnlp-emotion",
    "google-bert-bert-base-multilingual-cased": "multi-BERT",
    "cardiffnlp-hate": "cardiffnlp-hate",
    "cardiffnlp-offensive": "cardiffnlp-offensive"
}

class GroupModelScores:
    """
    Class to group score files for a given model, across domains and LLMs.
    """
    def __init__(self, model_name, template, prompt, domains, llm_models, experiment_name=""):
        self.model_name = model_name
        self.template = template
        self.prompt = prompt
        self.domains = domains
        self.llm_models = llm_models
        self.experiment_name = experiment_name
        self.clean_model_name = model_name_map[model_name]

        self.df_score_grouped = None
        self.grouped_score_path = None

    def group(self):
        dfs = []
        links = self._get_llm_links()
        noes_links = self._get_noes_links()
        links += noes_links

        for link in links:
            df = self._process_score_file(link, noes_links)
            dfs.append(df)

        # Process original template
        original_template_path = f"Scores/{self.model_name}/benchmark/{self.experiment_name}/scores_{self.template}.csv"
        for domain in self.domains:
            df = pd.read_csv(original_template_path)
            df = clean_unnamed(df)
            df["template_type"] = self.template
            df["model"] = self.clean_model_name
            df["domain"] = domain
            dfs.append(df)

        # Final aggregation
        self.df_score_grouped = pd.concat(dfs, axis=0)
        self.grouped_score_path = f"Scores/{self.model_name}/{self.template}/{self.prompt}/{self.experiment_name}/scores_all.csv"
        self.df_score_grouped.to_csv(self.grouped_score_path, index=False)

    def _get_llm_links(self):
        links = []
        for llm in self.llm_models:
            for domain in self.domains:
                path = f"Scores/{self.model_name}/{self.template}/{self.prompt}/{self.experiment_name}/scores_{llm}_{domain}.csv"
                links.append(path)
        return links

    def _get_noes_links(self):
        noes_links = []
        for domain in self.domains:
            if "wiki" not in domain:
                noes_links.append(f"Scores/{self.model_name}/benchmark/{self.experiment_name}/scores_{domain_dict[domain]}.csv")
            else:
                if self.template == "IPTTS":
                    noes_links.append(self._balance_wikipedia_scores())
                else:
                    noes_links.append(f"Scores/{self.model_name}/benchmark/{self.experiment_name}/scores_{domain_dict['wikipedia_talks']}.csv")
        return noes_links

    def _balance_wikipedia_scores(self):
        path = f"Scores/{self.model_name}/benchmark/{self.experiment_name}"
        df_non = pd.read_csv(f"{path}/scores_wikipedia_talks_nontoxic.csv")
        df_tox = pd.read_csv(f"{path}/scores_wikipedia_talks_toxic.csv")

        n_min = min(df_non["template_index"].max(), df_tox["template_index"].max())
        df_non = df_non[df_non["template_index"].isin(range(n_min))]
        df_tox = df_tox[df_tox["template_index"].isin(range(n_min))]
        df_tox["template_index"] += n_min + 1

        df = pd.concat([df_non, df_tox]).reset_index(drop=True)
        output_path = f"{path}/scores_wikipedia_talks_equilibrado.csv"
        df.to_csv(output_path, index=False)
        return output_path

    def _process_score_file(self, path, noes_paths):
        df = pd.read_csv(path)
        df = clean_unnamed(df)
        df["model"] = self.clean_model_name

        # Identify LLM name from filename
        name_start = path.index("scores_") + len("scores_")
        model_llm_name = path[name_start:].replace(".csv", "")
        for suffix in ["_tweets", "_wikipedia_talks", "_IMBD"]:
            model_llm_name = model_llm_name.replace(suffix, "")
        if "llama" in path or "mixtral" in path:
            model_llm_name += f"_{self.prompt}"

        df["template_type"] = model_llm_name

        for domain in self.domains:
            if domain in path:
                df["domain"] = domain
        if path in noes_paths:
            df["template_type"] = df["domain"]  
        return df
    


class ScoreAggregator:
    def __init__(self, path_to_scores: str):
        """
        Loads the score CSV and calculates a normalized 'score' column 
        based on the available logits.
        """
        self.df_score = pd.read_csv(path_to_scores)
        self._compute_score()

        # Keep only the relevant columns
        self.df_score = self.df_score[
            ["domain", "model", "template", "template_type", "group", "template_index", "score"]
        ].copy()

        self.df_background = None
        self.df_bias_group_noes = None
        self.df_bias_group_others = None

    def _compute_score(self):
        """Compute and normalize the score based on logits present."""
        cols = self.df_score.columns
        if {"0", "1", "2", "3"}.issubset(cols):
            score = self.df_score["0"] + self.df_score["1"] - self.df_score["2"] - self.df_score["3"]
        elif {"0", "2"}.issubset(cols):
            score = self.df_score["2"] - self.df_score["0"]
        else:
            score = self.df_score["1"]

        # Normalize
        min_val, max_val = score.min(), score.max()
        if max_val>1 or min_val<0:
            self.df_score["score"] = (score - min_val) / (max_val - min_val)
        else:
            self.df_score["score"] = (score)

    def get_template_map(self, template_type: str, domain: str) -> dict:

        """
        Returns a dictionary mapping template_index to template string
        for a specific template type and domain.
        """
        df = self.df_score[
            (self.df_score["template_type"] == template_type) &
            (self.df_score["domain"] == domain)
        ]
        max_idx = df["template_index"].max()
        df = df[df["template_index"] < max_idx]
        return dict(zip(df["template_index"], df["template"]))

    def find_intersections(self, domain: str):
        """
        Find shared differing template indices between 'EEC' and other LLM templates.
        Returns a summary DataFrame and a list of shared indices.
        """
        eec_templates = self.get_template_map("EEC", domain)
        unique_templates = self.df_score["template_type"].unique()
        summary = {"LLM": [], "different": []}
        shared_indices = set(eec_templates.keys())
        difference_dict = {}

        for template_type in unique_templates:
            if template_type in ["tweets", "wikipedia_talks", "IMBD", "EEC"]:
                continue
            comp_templates = self.get_template_map(template_type, domain)
            differing = [idx for idx in comp_templates if comp_templates[idx] != eec_templates.get(idx)]
            summary["LLM"].append(template_type)
            summary["different"].append(len(differing))
            difference_dict[template_type] = differing
            shared_indices &= set(differing)

        return pd.DataFrame(summary), list(shared_indices)

    def filter_templates(self):
        """
        Filters the score DataFrame to only keep templates with differences
        compared to EEC, plus the baseline NOE templates.
        """
        filtered = []
        domains = self.df_score["domain"].unique()

        for domain in domains:
            _, differing_indices = self.find_intersections(domain)
            llm_rows = self.df_score[
                (self.df_score["domain"] == domain) &
                (self.df_score["template_index"].isin(differing_indices)) &
                (~self.df_score["template_type"].isin(["tweets", "wikipedia_talks", "IMBD"]))
            ]
            noe_rows = self.df_score[
                (self.df_score["domain"] == domain) &
                (self.df_score["template_type"].isin(["tweets", "wikipedia_talks", "IMBD"]))
            ]
            filtered.extend([llm_rows, noe_rows])

        self.df_score = pd.concat(filtered, axis=0).reset_index(drop=True)

    def compute_background_scores(self, specific_group: str = None):
        """
        Computes the background scores for each template index.
        If specific_group is given, only that group's scores are used.
        """
        if specific_group:
            df = self.df_score[self.df_score["group"] == specific_group]
        else:
            df = self.df_score

        group_cols = ["domain", "model", "template_index", "template_type"]
        self.df_background = (
            df[group_cols + ["score"]]
            .groupby(group_cols)
            .mean()
            .reset_index()
            .rename(columns={"score": "score_background"})
        )

    def compute_DP(self):
        """
        Calculates the DP (Delta Privacy) score as:
        DP = 1 - |score - score_background|
        """
        if self.df_background is None:
            raise ValueError("Background scores not computed.")

        self.df_score = self.df_score.merge(
            self.df_background,
            on=["template_index", "template_type", "domain", "model"],
            how="left"
        )
        self.df_score["DP"] = 1 - np.abs(self.df_score["score"] - self.df_score["score_background"])

    def group_DP_by_template(self):
        """
        Groups DP scores by template type and distinguishes between:
        - NOEs (named after domains)
        - Other LLM-generated templates
        """
        domain_names = self.df_score["domain"].unique().tolist()
        is_noe = self.df_score["template_type"].isin(domain_names)

        self.df_bias_group_noes = (
            self.df_score[is_noe]
            .groupby(["domain", "model", "group", "template_type"])[["DP"]]
            .mean()
            .reset_index()
        )

        self.df_bias_group_others = (
            self.df_score[~is_noe]
            .groupby(["domain", "model", "group", "template_type"])[["DP"]]
            .mean()
            .reset_index()
        )


class GroupModelAggregator:
    def __init__(self, model_names, template, prompt, domains, llm_models, experiment_name=""):
        self.model_names = model_names
        self.prompt = prompt
        self.template = template
        self.domains = domains
        self.llm_models = llm_models
        self.experiment_name = experiment_name

        self.df_vbcm_noes = None
        self.df_vbcm_others = None
        self.mae_scores = None
        self.pearson_scores = None

    def _reshape_metrics_by_domain(self, df_bias_metrics, metric_name):
        """Pivot the metrics (MAE or Pearson) to have a domain-wise view per model/template."""
        df = df_bias_metrics[["domain", "model", "template_type", metric_name]]
        domain_dfs = []

        for domain in self.domains:
            df_domain = df[df["domain"] == domain].copy()
            df_domain = df_domain.rename(columns={metric_name: f"{metric_name}_{domain}"})
            df_domain = df_domain.drop(columns=["domain"])
            domain_dfs.append(df_domain.reset_index(drop=True))

        merged_df = domain_dfs[0]
        for next_df in domain_dfs[1:]:
            merged_df = merged_df.merge(next_df, on=["model", "template_type"], how="inner")

        merged_df["template_type"] = merged_df["template_type"].apply(
            lambda x: x.replace(f"llama3_70_{self.prompt}", f"{self.template}-LLaMa3-70B")
                      .replace(f"llama3_8_{self.prompt}", f"{self.template}-LLaMa3-8B")
                      .replace(f"mixtral_{self.prompt}", f"{self.template}-Mixtral8x7B")
        )

        return merged_df

    def compute_group_scores(self):
        """Compute and save grouped DP scores for each model."""
        noes_list = []
        others_list = []

        for model in self.model_names:
            print(f"Processing model: {model}")
            scorer = GroupModelScores(model, self.template, self.prompt, self.domains, self.llm_models, self.experiment_name)
            scorer.group()

            print(f"Grouped score path created: {scorer.grouped_score_path}")
            aggregator = ScoreAggregator(scorer.grouped_score_path)
            aggregator.compute_background_scores()
            aggregator.compute_DP()
            aggregator.group_DP_by_template()

            print("VBCM scores computed for grouped model.")
            noes_list.append(aggregator.df_bias_group_noes)
            others_list.append(aggregator.df_bias_group_others)

        self.df_vbcm_noes = pd.concat(noes_list).reset_index(drop=True)
        self.df_vbcm_others = pd.concat(others_list).reset_index(drop=True)

        self.df_vbcm_noes.to_csv(f"Scores/dataframes/{self.template}_{self.experiment_name}_{self.prompt}_NOEs.csv", index=False)
        self.df_vbcm_others.to_csv(f"Scores/dataframes/{self.template}_{self.experiment_name}_{self.prompt}_Others.csv", index=False)

        df_all = pd.concat([self.df_vbcm_others, self.df_vbcm_noes])
        df_all.to_csv(f"Scores/dataframes/{self.template}_{self.experiment_name}_{self.prompt}_ALL.csv", index=False)

    def load_vbcm_scores(self, path_noes, path_others):
        """Load precomputed VBCM scores from file."""
        self.df_vbcm_noes = pd.read_csv(path_noes)
        self.df_vbcm_others = pd.read_csv(path_others)

    def compute_metrics(self):
        """Compute MAE and Pearson metrics by comparing NOEs and other groups."""
        merged = self.df_vbcm_others.merge(
            self.df_vbcm_noes,
            on=["domain", "model", "group"],
            how="left",
            suffixes=("", "_noes")
        )
        merged["MAE"] = (merged["DP"] - merged["DP_noes"]).abs()

        mae_summary = merged.groupby(["domain", "model", "template_type"])["MAE"].mean().reset_index()
        pearson_summary = merged.groupby(["domain", "model", "template_type"])["DP"].corr(merged["DP_noes"]).reset_index()
        pearson_summary = pearson_summary.rename(columns={"DP": "Pearson"})

        metric_summary = mae_summary.merge(pearson_summary, on=["domain", "model", "template_type"])

        self.mae_scores = self._reshape_metrics_by_domain(metric_summary, "MAE")
        self.pearson_scores = self._reshape_metrics_by_domain(metric_summary, "Pearson")



class ScorePlotter:
    def __init__(self, template, experiment_name, prompt, mae_scores, pearson_scores):
        self.template = template
        self.experiment_name = experiment_name
        self.prompt = prompt
        self.mae_scores = mae_scores
        self.pearson_scores = pearson_scores

        self.models = []
        self.template_types = []
        self.domains = []

    def load_scores(self, path_mae, path_pearson):
        """Load scores from CSV files."""
        self.mae_scores = pd.read_csv(path_mae)
        self.pearson_scores = pd.read_csv(path_pearson)

    def _extract_plot_metadata(self, metric="MAE"):
        """Prepare labels, orders and colors for plotting."""
        df = self.mae_scores if metric == "MAE" else self.pearson_scores

        self.models = df["model"].unique().tolist()
        self.template_types = df["template_type"].unique().tolist()
        self.domains = [col.replace(f"{metric}_", "") for col in df.columns if col.startswith(metric)]

        color_palette = {
            f'{self.template}': "#4768ed",
            f'{self.template}-LLaMa3-8B': "#d6d9b6",
            f'{self.template}-LLaMa3-70B': "#c3cc62",
            f'{self.template}-Mixtral8x7B': "#b8c714",
            'NOEs': "#798406"
        }

        domain_labels = {
            "wikipedia_talks": "Wikipedia Talks Pages",
            "tweets": "Tweets",
            "IMBD": "IMDB"
        }

        label_map = {
            "EEC": {
                'distil-BERT': 'DistilBERT\n(FT)',
                'BERT': 'BERT\n(FT)',
                'cardiffnlp-roberta-xlm': 'cardiffnlp-twitter-XLM\nbase (FT)',
                'cardiffnlp-sentiment': 'cardiffnlp-sentiment\n(off the shelf)',
                'cardiffnlp-emotion': 'cardiffnlp-emotion\n(off the shelf)'
            },
            "IPTTS": {
                'distil-BERT': 'DistilBERT\n(FT)',
                'BERT': 'BERT\n(FT)',
                'multi-BERT': 'multi-BERT\nbase (FT)',
                'cardiffnlp-hate': 'cardiffnlp-hate\n(off the shelf)',
                'cardiffnlp-offensive': 'cardiffnlp-offensive\n(off the shelf)'
            }
        }

        model_order = [m for m in label_map[self.template] if m in self.models]
        model_labels = [label_map[self.template][m] for m in model_order]
        template_order = [f'{self.template}', f'{self.template}-LLaMa3-8B', f'{self.template}-LLaMa3-70B', f'{self.template}-Mixtral8x7B']

        return model_order, model_labels, template_order, domain_labels, color_palette

    def plot_bar_charts(self, metric="MAE"):
        """Plot bar charts for the given metric (MAE or Pearson)."""
        df = self.mae_scores if metric == "MAE" else self.pearson_scores
        model_order, model_labels, template_order, domain_labels, palette = self._extract_plot_metadata(metric)

        fig, axes = plt.subplots(len(self.domains), 1, figsize=(18, 9), sharex=True)

        for i, domain in enumerate(self.domains):
            sns.barplot(
                data=df,
                x='model',
                y=f'{metric}_{domain}',
                hue='template_type',
                ax=axes[i],
                errorbar=None,
                palette=palette,
                order=model_order,
                hue_order=template_order,
                width=0.6,
            )

            axes[i].set_ylabel(metric, fontsize=18)
            axes[i].set_title(f"$\\mathcal{{D}}$ = {domain_labels[domain]}", loc="left", fontsize=22, fontweight='semibold')
            axes[i].set_xticklabels(model_labels, rotation=0, fontsize=14)
            axes[i].grid(True, linestyle='--', axis='y', alpha=0.8)

            if metric == "Pearson":
                axes[i].set_ylim(0, df[f"{metric}_{domain}"].max() + 0.01)
            else:
                axes[i].set_ylim(0, df[f"{metric}_{domain}"].max() + 1e-5)

            if i < len(self.domains) - 1:
                axes[i].legend_.remove()
            else:
                axes[i].legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.4), fontsize=22)

        plt.tight_layout()
        output_path = f'/home/tquiroga/llm_test_1/Adaptation/graphs/clean/{metric}_{self.template}_{self.experiment_name}_{self.prompt}_BARPLOT.pdf'
        plt.savefig(output_path, bbox_inches='tight')
        plt.show()

    def plot_scatter(self):
        """To be implemented: plot scatter comparisons if needed."""
        pass
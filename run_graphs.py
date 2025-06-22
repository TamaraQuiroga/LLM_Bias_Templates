from Clean_Code.bias_metrics_and_plots import GroupModelAggregator
from Clean_Code.bias_metrics_and_plots import ScorePlotter

# Experiment configuration
prompt = "f3"
template = "EEC"
experiment_name = ""

domains = ["tweets", "wikipedia_talks", "IMBD"]
llm_models = ["llama3_8", "llama3_70", "mixtral"]
fine_tuned_models = ["bert-base-cased", "distilbert-distilbert-base-cased", "cardiffnlp-sentiment"]

# Aggregate model results
aggregator = GroupModelAggregator(
    fine_tuned_models,
    template,
    prompt,
    domains,
    llm_models,
    experiment_name
)

# Load precomputed bias scores (NOEs and others)
aggregator.load_vbcm_scores(
    f"Scores/dataframes/{template}_{experiment_name}_{prompt}_NOEs.csv",
    f"Scores/dataframes/{template}_{experiment_name}_{prompt}_Others.csv"
)

# Compute MAE and Pearson metrics
aggregator.compute_metrics()
mae_scores = aggregator.mae_scores
pearson_scores = aggregator.pearson_scores

# Plot results
plotter = ScorePlotter(
    template,
    experiment_name,
    prompt,
    mae_scores,
    pearson_scores
)

plotter.plot_bar_charts(metric="MAE")
plotter.plot_bar_charts(metric="Pearson")
print("Plots generated successfully.")
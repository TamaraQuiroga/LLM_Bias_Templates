from Classgraphs import AgruparGrupoModelos
from Classgraphs import PlotScores

prompt = "f3"
template = "EEC"
name_experiment = ""
l_domains = ["tweets","wikipedia_talks","IMBD"]
l_llm_models = ["llama3_8","llama3_70","mixtral"]
l_models = ["bert-base-cased","cardiffnlp-sentiment"]

# Comprimir resultados
experiment_results = AgruparGrupoModelos(l_models,
                                         template,
                                         prompt,
                                         l_domains,
                                         l_llm_models,
                                         name_experiment)
experiment_results.agrupar_scores_df()

MAE_score = experiment_results.score_MAE_models
Pearson_score = experiment_results.score_Pearson_models

# Graficar
graphs = PlotScores(template,
                    name_experiment,
                    prompt,
                    MAE_score,
                    Pearson_score)
graphs.plot_bar(var="MAE")
graphs.plot_bar(var="Pearson")
print("graficos listos")
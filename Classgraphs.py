import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def MAE(a, b):
    v = np.mean(np.abs(a-b))
    return v
def MSE( a, b):
    v = np.mean((a-b)**2)
    return v

def clean_unnamed(df):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.reset_index(drop=True) 
    return df  

dict_domain = {"tweets":"tweets","wikipedia_talks":"wikipedia_talks_nontoxic","IMBD":"IMBD"}
standar_names_model = {
                "bert-base-cased": "BERT",
                "distilbert-distilbert-base-cased": "distil-BERT",
                "cardiffnlp-twitter-xlm-roberta-base":"cardiffnlp-roberta-xlm",
                "cardiffnlp-sentiment":"cardiffnlp-sentiment",
                "cardiffnlp-emotion":"cardiffnlp-emotion",
                "google-bert-bert-base-multilingual-cased":"multi-BERT",
                "cardiffnlp-hate":"cardiffnlp-hate",
                "cardiffnlp-offensive":"cardiffnlp-offensive"}


class agrupar_scores_model:
    def __init__(self,name_model,template, prompt, l_domains, l_llm_models, name_experiment= ""):
        self.name_model = name_model
        self.prompt = prompt
        self.template = template
        self.l_domains = l_domains
        self.l_llm_models = l_llm_models
        self.name_experiment = name_experiment
        self.clean_name_model = standar_names_model[self.name_model]

        self.path_score_agrupado = None
        self.df_score_agrupado = None
    
    def agrupar(self):
        l_dfs = []
        links = []

        # LLM-Tempaltes links
        for model_llm in self.l_models_llm :
            for domain_ in self.l_domains:
                link_ = f"Scores/{self.name_model}/{self.template}/{self.prompt}/{self.name_experiment}/scores_{model_llm}_{domain_}.csv"
                links.append(link_)

        # NOES linkks
        data_noes = []
        for domain_i in self.l_domains:
            if "wiki" not in domain_i:
                data_noes += [f"Scores/{self.name_model}/benchmark/{self.name_experiment}/scores_{dict_domain[domain_i]}.csv"]
            else:
                if self.template =="IPTTS":
                    df_nontoxic =pd.read_csv(f"Scores/{self.name_model}/benchmark/{self.name_experiment}/scores_wikipedia_talks_nontoxic.csv")
                    df_toxic =pd.read_csv(f"Scores/{self.name_model}/benchmark/{self.name_experiment}/scores_wikipedia_talks_toxic.csv")
                    
                    n_min = min(int(max(df_toxic["template_index"].unique())),int(max(df_nontoxic["template_index"].unique())))
                    
                    df_nontoxic = df_nontoxic[df_nontoxic["template_index"].isin(range(n_min))]
                    df_toxic =  df_toxic[df_toxic["template_index"].isin(range(n_min))]
                    df_toxic["template_index"] =  df_toxic["template_index"].apply(lambda x: x+n_min+1)
                    df = pd.concat([df_nontoxic, df_toxic])
                    df = df.reset_index(drop=True)
                    df.to_csv(f"Scores/{self.name_model}/benchmark/{self.name_experiment}/scores_wikipedia_talks_equilibrado.csv", index=False)
                    data_noes += [f"Scores/{self.name_model}/benchmark/{self.name_experiment}/scores_wikipedia_talks_equilibrado.csv"]
                else:
                    data_noes += [f"Scores/{self.name_model}/benchmark/{self.name_experiment}/scores_{dict_domain["wikipedia_talks"]}.csv"]
        links +=data_noes

        # Original templates Links
        original_template_link = [f"Scores/{self.model_original}/benchmark/{self.name_experiment}/scores_{self.template}.csv"]

        
        # Scores NOES y LLM - templates
        for link in links:
            df = pd.read_csv(link)
            df = clean_unnamed(df)
            df["model"] = self.clean_name_model
            index_word = link.index("scores_")
            len_word = len("scores_")
            if "llama" in link or "mixtral" in link:
                model_llm_link = link[index_word+len_word:].replace("_tweets.csv","").replace("_wikipedia_talks.csv","").replace("_IMBD.csv","")+"_f3"
                model_llm_link = model_llm_link.replace(".csv","")
            else:
                model_llm_link = link[index_word+len_word:].replace(".csv","")
            print(model_llm_link)
            df["template_type"] =model_llm_link
            for domain_i in self.l_domains:
                if domain_i in link:
                    df["domain"] = domain_i
            if link in data_noes:
                df["template_type"] = df["domain"]
            l_dfs.append(df)

        # Scores EEC-IPTTS  
        for k in range(len(self.l_domains)):
            for link in original_template_link:
                df = pd.read_csv(link)
                df = clean_unnamed(df)
                df["template_type"] = f"{self.template}"
                df["model"] = self.clean_name_model
                df["domain"] =  self.l_domains[k]
                l_dfs.append(df)
    
        # Agrupar todo socres
        self.df_score_agrupado  = pd.concat(l_dfs, axis = 0)
        self.path_score_agrupado = f"Scores/{self.model_original}/{self.template}/{self.prompt}/{self.name_experiment}/scores_all.csv"
        self.df_score_agrupado.to_csv(self.path_score_agrupado,index=False)

     
class ScoreSum:
    def __init__(self,path_df):
        
        self.df_score = pd.read_csv(path_df)
        print(self.df_score.shape)

        if "2" in  self.df_score.columns and "0" in  self.df_score.columns and "3"  in self.df_score.columns :
            self.df_score["score"] = self.df_score.apply(lambda x : x["0"]+x["1"]-x["2"]-x["3"], axis=1)
            max_value = self.df_score["score"].max()
            min_value = self.df_score["score"].min()
            self.df_score["score"] = self.df_score["score"].apply(lambda x: (x-min_value)/(max_value-min_value))
        elif "2" in  self.df_score.columns and "0" in  self.df_score.columns:
            self.df_score["score"] = self.df_score.apply(lambda x : x["2"]-x["0"], axis=1)
            max_value = self.df_score["score"].max()
            min_value = self.df_score["score"].min()
            self.df_score["score"] = self.df_score["score"].apply(lambda x: (x-min_value)/(max_value-min_value))
        else:
            self.df_score["score"] = self.df_score.apply(lambda x : x["score"], axis=1)

            
        self.df_score = self.df_score[["domain","model","template","template_type","group","template_index","score"]]        
        self.df_background = None
        
        self.gender = "male"
        self.type_bias = "nationality" 
        
        self.df_bias_group_NOEs = None
        self.df_bias_group_others = None


    def template_llm(self,type_llm,domain):
        df_template_EEC = self.df_score[(self.df_score["template_type"]==type_llm)&(self.df_score["domain"]==domain)]
        n_max = df_template_EEC["template_index"].max()
        
        df_template_EEC = df_template_EEC[:n_max]
        df_template_EEC = df_template_EEC[["template","template_index"]]
        template_index = df_template_EEC["template_index"].tolist()
        template = df_template_EEC["template"].tolist()
        dict_EEC = {template_index[i]:template[i] for i in range(len(template))}
        return dict_EEC
    
    def inter_list(self,domain):
        EEC = self.template_llm("EEC",domain)
        all_templates_type = self.df_score["template_type"].unique().tolist()

        dict_df = {"LLM":[],"different":[]}
        inter_list = [i for i  in EEC]
        dict_inter = {}
        for template_i in all_templates_type:
            if template_i not in ["tweets","wikipedia_talks","IMBD","EEC"]:
                dict_template_i = self.template_llm(template_i,domain)
                L = []
                for i in dict_template_i:
                    if dict_template_i[i]!=EEC[i]:
                        L.append(i)
                dict_df["LLM"].append(template_i)
                dict_df["different"].append(len(list(L)))
                dict_inter[template_i] = L
                inter_list = list(set(inter_list)& set(L))
        return pd.DataFrame(dict_df),inter_list

    def filter_df_score(self):
        l_domain = self.df_score["domain"].unique().tolist()
        L_all = []
        for domain_i in l_domain:
            l = self.inter_list(domain_i)[1]
            print(domain_i)
            print(len(l))
            llm_list = self.df_score[(self.df_score["domain"]==domain_i)&( self.df_score["template_index"].isin(l))&(~self.df_score["template_index"].isin(["tweets","wikipedia_talks","IMBD"]))]
            not_llm_list = self.df_score[(self.df_score["domain"]==domain_i)&self.df_score["template_index"].isin(["tweets","wikipedia_talks","IMBD"])]
            L_all.append(llm_list)
            L_all.append(not_llm_list)

        self.df_score = pd.concat(L_all, axis = 0)
        self.df_score = self.df_score.reset_index(drop=True)


    def df_background_cal(self):
        df_mean = self.df_score[["domain","model","template_index","template_type","score"]].groupby(["domain","model","template_index","template_type"]).mean()
        df_mean = df_mean.reset_index()
        self.df_background = df_mean.rename(columns={"score":"score_background"})
        print("self.df_background",self.df_background.shape)

    def df_background_cal_specific(self,g_name="M-white"):
            df_score_specific = self.df_score[self.df_score["group"]==g_name]
            df_mean = df_score_specific[["domain","model","template_index","template_type","score"]].groupby(["domain","model","template_index","template_type"]).mean()
            df_mean = df_mean.reset_index()
            self.df_background = df_mean.rename(columns={"score":"score_background"})
            print("self.df_background",self.df_background.shape)

    def df_score_DP(self):
        print("self.df_score",self.df_score.shape)
        self.df_score = self.df_score.merge(self.df_background, on = ["template_index","template_type","domain","model"], how="left")
        print("self.df_score",self.df_score.shape)

        self.df_score = self.df_score.reset_index(drop = True)
        self.df_score["DP"] = self.df_score.apply(lambda x: 1-np.abs(x["score"]-x["1_background"]), axis=1)

    def df_DP_group(self):
        domains = self.df_score["domain"].unique().tolist()
        df_aux0 = self.df_score.copy()

        df_aux = df_aux0[df_aux0["template_type"].isin(domains)]
        df_aux = df_aux.reset_index(drop=True)
        self.df_bias_group_NOEs = df_aux[["group","DP","template_type","domain","model"]].groupby(["domain","model","group","template_type"]).mean() #agrupar nivel de tempalte_id
        self.df_bias_group_NOEs = self.df_bias_group_NOEs.reset_index()
        
        df_aux = df_aux0[~df_aux0["template_type"].isin(domains)]
        df_aux = df_aux.reset_index(drop=True)
        self.df_bias_group_others = df_aux[["group","DP","template_type","domain","model"]].groupby(["domain","model","group","template_type"]).mean() #agrupar nivel de tempalte_id
        self.df_bias_group_others = self.df_bias_group_others.reset_index()



class AgruparGrupoModelos:
    def __init__(self,l_name_models,template, prompt, l_domains, l_llm_models, name_experiment= ""):
        self.l_name_models = l_name_models
        self.prompt = prompt
        self.template = template
        self.l_domains = l_domains
        self.l_llm_models = l_llm_models
        self.name_experiment = name_experiment
    
        self.score_MAE_models = None
        self.score_Pearson_models = None
    def metrics_cal(self,df_bias_metrics,var):
        # var = "MAE"
        df_sum_MAE = df_bias_metrics[["domain","model","template_type",var]]
        dict_MAE = {}
        for domain_ in self.l_domains:
            df_sum_MAE_i=df_sum_MAE[df_sum_MAE["domain"]==domain_].reset_index(drop=True)
            df_sum_MAE_i = df_sum_MAE_i.rename(columns={var:var+"_"+domain_}).drop(columns=["domain"])
            dict_MAE[domain_] = df_sum_MAE_i

        df_MAE = dict_MAE[self.l_domains[0]]
        for i in range(len(self.l_domains[1:])):
            df_MAE = df_MAE.merge(dict_MAE[self.l_domains[1:][i]],on=["model","template_type"],how="inner")
        df_MAE["template_type"] = df_MAE["template_type"].apply(lambda x: x.replace(f"llama3_70_{self.prompt}",f"{self.template}-LLaMa3-70B").
                                                                                replace(f"llama3_8_{self.prompt}",f"{self.template}-LLaMa3-8B").
                                                                                replace(f"mixtral_{self.prompt}",f"{self.template}-Mixtral8x7B"))

        return df_MAE
        
    def agrupar_scores_df(self):
        L_df_NOEs = []
        L_df= []

        for model_i in self.l_name_models:
            model_socres = agrupar_scores_model(model_i,self.prompt,self.template,self.l_domains,self.l_llm_models,self.name_experiment)
            model_socres.agrupar()
            templates = ScoreSum(model_socres.path_score_agrupado)
            templates.df_background_cal()
            templates.df_score_DP()
            templates.df_DP_group()
            L_df.append(templates.df_bias_gruop_others)
            L_df_NOEs.append(templates.df_bias_gruop_NOEs)
        
        df_NOEs = pd.concat(L_df_NOEs, axis=0)
        df_NOEs = df_NOEs.reset_index(drop=True)
        df_NOEs.to_csv(f"Scores/dataframes/{self.template}_{self.name_experiment}_{self.prompt}_NOEs.csv",index=False)

        df = pd.concat(L_df, axis=0)
        df = df.reset_index(drop=True)
        df.to_csv(f"Scores/dataframes/{self.template}_{self.name_experiment}_{self.prompt}_Others.csv",index=False)

        df_all = pd.concat([df,df_NOEs], axis=0)
        df_all.to_csv(f"Scores/dataframes/{self.template}_{self.name_experiment}_{self.prompt}_ALL.csv",index=False)

 
        df_bias_metrics = df.merge(df_NOEs, on =["domain","model","group"],how="left",suffixes=("","_0")) 
        df_bias_metrics["MAE"] = df_bias_metrics.apply(lambda x : abs(x["DP"]-x["DP_0"]), axis=1)
        
        df_MAE_cal = df_bias_metrics.groupby(["domain","model","template_type"])["MAE"].mean().reset_index()
        
        df_PEARSON_cal = df_bias_metrics.groupby(["domain","model","template_type"])['DP'].corr(df['DP_0']).reset_index()
        df_PEARSON_cal = df_PEARSON_cal.rename(columns={"DP":"Pearson"})
        df_bias_metrics = df_MAE_cal.merge(df_PEARSON_cal, on = ["domain","model","template_type"])


        df_finalMAE = self.metrics_cal(self,df_bias_metrics,"MAE")
        df_finalPearson = self.metrics_cal(self,df_bias_metrics,"Pearson")

        
        self.score_MAE_models = df_finalMAE
        self.score_Pearson_models = df_finalPearson

class PlotScores:
    def __init__(self):
        pass
    
    def MAE_plot(self):
        pass
        

    def Pearson_plot(self):
        pass
        

    def Scatter_plot(self):
        pass

import gc
import time
import torch
from pathlib import Path
from adapted_template_llm import AdaptedTemplateLLM
from pertubations import PerturbationExperiment



def main():
    # === CONFIGURATION ===
    templates = {
        "EEC": "Templates/EEC_counterfactual.csv",
        "IPTTS": "Templates/IPTTS_counterfactual.csv"
    }

    domains = {
        "tweets": "Domain_Examples/tweets_domain_examples.csv",
        "wikipedia_talks": "Domain_Examples/wikipedia_talks_domain_examples.csv",
        "IMBD": "Domain_Examples/IMDB_domain_examples.csv"
    }

    model_list = ["llama3_8"]  # Add more models as needed
    domain_list = ["tweets"]
    template_name = "EEC"
    prompt_id = "f3"
    experiment_name = ""
    n_examples = 15

    # === OUTPUT TIME TRACKING FILE ===
    time_path = Path("Check/time_llm_time_few_shot_zero.csv")
    time_path.parent.mkdir(exist_ok=True)
    time_path.write_text("")  # Clear previous

    # === MAIN LOOP ===
    for model_name in model_list:
        for domain_name in domain_list:
            start_time = time.time()
            print(f"Running for: Template={template_name}, Model={model_name}, Domain={domain_name}")

            # File paths
            path_template = templates[template_name]
            path_domain = domains[domain_name]

            # Create experiment object
            experiment = AdaptedTemplateLLM(
                name_template=template_name,
                name_domain=domain_name,
                model_name=model_name,
                prompt_name=prompt_id,
                path_template=path_template,
                path_domain_examples=path_domain,
                experiment_name=experiment_name,
                few_examples=False
            )

            # Prepare folders
            output_dir = Path(f"LLM_templates/originales/{template_name}/{prompt_id}/{experiment_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load and prepare data
            experiment.load_template_df()
            experiment.load_prompt_text()
            experiment.load_domain_examples()
            experiment.replace_names_with_top()

            # Generate model responses
            experiment.generate_template(n_examples=n_examples)

            # Save updated template with responses
            experiment.df_template.to_csv(f"LLM_templates/originales/{template_name}/{prompt_id}/{experiment_name}/llm_template_{model_name}_{domain_name}.csv", index=False)
                #data.df_template.to_csv(f"adaptation_llm/originales/{template}/{prompt_id}/{name_experiment}/llm_template_{model_name}_{domain}.csv") # PATH        

            # Log execution time
            duration = time.time() - start_time
            with open(time_path, "a") as log_file:
                log_file.write(f"{model_name}, {domain_name}, {template_name}, {prompt_id}, {duration:.2f}\n")


            # Pertubation LLM-template
            pertubation_experiment = PerturbationExperiment(template_name,
                                                            prompt_id,
                                                            model_name,
                                                            domain_name,
                                                            experiment=experiment_name,
                                                            group_json_path="Group_Identity_Terms/nationality_names_50.json")
            pertubation_experiment.run()

            # Pertubation Original-template
            pertubation_experiment.run_original_template()
            # Pertubation NOes-tmeplate
            pertubation_experiment.run_NOEs_template()


            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Completed in {duration:.2f} seconds\n")

if __name__ == "__main__":
    main()

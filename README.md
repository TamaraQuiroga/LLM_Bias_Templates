# LLM_Bias_Templates

# Domain-Adaptive Bias Evaluation with LLM-Prompted Template Generation

This repository contains code and data related to our work on adapting social bias evaluation datasets to specific domains using Large Language Models (LLMs).

## Overview

Many datasets exist to measure social bias in NLP systems, but evaluating bias in domain-specific contexts remains challenging. This project introduces a domain-adaptive framework that uses LLM prompting to automatically convert template-based bias datasets into domain-specific versions.

We apply this approach to two popular benchmarks:
- **Equity Evaluation Corpus (EEC)**
- **Identity Phrase Templates Test Set (IPTTS)**

The datasets are adapted to the Twitter and Wikipedia Talk domains. Our experiments show that the adapted datasets provide bias estimates that better reflect real-world data.

## Contents

- `adapted_template_llm.py` - scripts for LLM prompting and create adapted bias datasets 
- `eval_ftmodels.py` - evaluate finetuning models in  adapted bias datasets
- `eval_offtheshelf.py` - evaluate off-the-shelf models in  adapted bias datasets
- `bias_metrics_and_plots.py` - exploratory analysis and visualization

## Usage

Instructions to run the adaptation and evaluation scripts will be added here soon.

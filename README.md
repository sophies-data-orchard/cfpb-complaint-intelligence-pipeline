# CFPB Complaint Intelligence Pipeline

An end-to-end NLP/ML pipeline built on the CFPB Consumer Complaint Database — from raw bulk data through EDA, topic modeling, entity extraction, classification, retrieval, and an LLM-optimized triage tool. Six notebooks, each building on the last, scoped to credit reporting complaints (the largest single category in the dataset for the analyzed window).

**The throughline isn't just "here's a model that works."** Several steps here turned up null results, inflated metrics, and biased evaluations — and the project treats catching and reporting those honestly as part of the work, not a detour from it. See each notebook's Summary/Takeaway section for what held up and what didn't.

## Pipeline Overview

| # | Notebook | What it does | Key result |
|---|---|---|---|
| 01 | `01_eda.ipynb` | Scopes the raw 13.8M-row CFPB export down to a working slice; diagnoses a mixed-date-format parsing bug; quantifies near-duplicate narratives | 400,445 credit reporting complaints retained, 48.6% of narratives found to be exact duplicates |
| 02 | `02_topic_modeling.ipynb` | Deduplicates narratives; fits LDA (k=7, chosen by coherence score); compares topics against CFPB's own `Issue` taxonomy | LDA topics partially validate, partially diverge from CFPB's categories |
| 03 | `03_ner_entity_extraction.ipynb` | Extracts dollar amounts, dates, and company mentions from free text via spaCy NER + regex; iterates through several failed company-name filters before landing on a working one | Row-level self-matching against each complaint's own `Company` field was the only filter that actually worked (26.5% clean match rate) |
| 04 | `04_classification_transformer_vs_distilled.ipynb` | Predicts company response to complaint across 5 baselines: structured fields, +topic labels, +entity features, fine-tuned DistilBERT, distilled bert-tiny | Diminishing returns past structured fields + one added feature set; transformer's edge is real but narrow and confounded by a smaller training sample |
| 05 | `05_rag_pipeline.ipynb` | Builds a FAISS + sentence-transformer retrieval system over historical complaints, validated against held-out queries | 0.86 mean issue-match rate on held-out retrieval queries |
| 06 | `06_dspy_triage_module.ipynb` | Uses DSPy to build and prompt-optimize a triage module (category, suggested approach, confidence) grounded in Notebook 05's retrieval | Optimization improved accuracy, but the evaluation metric itself needed three iterations before the result could be trusted |

## Data

This project uses the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — 13.8M+ public complaints about financial products and services, submitted since 2011, updated daily.

**As built:** the raw bulk CSV export was downloaded manually from the CFPB site and uploaded directly to Colab.

**Alternative — programmatic access:** the CFPB also provides a public [Search API](https://cfpb.github.io/ccdb5-api/documentation/) (no API key required) that supports the same filters as the website (date range, product, company, narrative presence, etc.) and returns results as JSON, CSV, XLS, or XLSX. For a reproducible pipeline, pulling directly from the API instead of a manual download is the better long-term approach and is a reasonable follow-up improvement to this repo — for example, `GET https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/` with query parameters for `date_received_min`, `date_received_max`, and `has_narrative=true`.

No data files are included in this repo (the raw export alone is several GB). Each notebook expects the relevant CSV(s) in a local `data/` directory, produced by the previous notebook in the sequence (e.g., Notebook 02 expects `data/01_complaints_credit_reporting.csv` from Notebook 01's output).


## Repo Structure

```
├── 01_eda.ipynb
├── 02_topic_modeling.ipynb
├── 03_ner_entity_extraction.ipynb
├── 04_classification_transformer_vs_distilled.ipynb
├── 05_rag_pipeline.ipynb
├── 06_dspy_triage_module.ipynb
├── requirements.txt
├── README.md
├── cfpb_pipeline_flowchart.png
└── data/                  # not included — populated by running notebooks in order
```

## How This Was Built

This project was built with Claude (Anthropic) as an AI coding and analysis assistant — most of the code and initial drafts of the analysis text were AI-generated. My role was directing the work: deciding what to investigate at each step, catching errors (including a few real bugs in the AI's own output — see notes below), setting the methodological standards (e.g., insisting on held-out evaluation splits, rejecting overstated claims, and rebuilding an evaluation metric twice after finding it didn't actually measure what it claimed to), running every diagnostic myself, and validating results against real output before drawing conclusions rather than accepting generated summaries at face value.


## Notes

- Each notebook is meant to be run in order — later notebooks load data artifacts saved by earlier ones.
- Notebook 04's transformer stages subsample the training set for compute reasons; this is stated explicitly in that notebook rather than left implicit, since it affects how those results compare to the non-subsampled baselines.
- Notebook 06's final evaluation metric went through several iterations after an initial version was found to reward vocabulary overlap rather than genuine category accuracy — that debugging process is left in the notebook rather than cleaned up, since it's as much a part of the result as the final number.

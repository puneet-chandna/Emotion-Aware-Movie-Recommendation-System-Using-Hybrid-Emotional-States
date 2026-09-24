# Emotion-Aware Movie Recommendation System Using Hybrid Emotional States

This repository contains an exploratory [Jupyter notebook](src_code.ipynb) that assigns basic and complex mood labels to movies using overview text, sentiment, and Sentence-BERT embeddings. It includes visualizations and an F1 calculation. The repository does not contain a deployed app, an interactive mood input screen, or an independently validated recommendation benchmark.

## Run locally

Use Python 3.11 or 3.12. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m notebook src_code.ipynb
```

The notebook reads the included `dataset.csv` and should be run in cell order with the repository root as the working directory. It processes the first 1,000 rows. Sentence-Transformers downloads `all-MiniLM-L6-v2` on first use, so the NLP cells require network access and enough time and memory for model inference. The spaCy command above downloads a separate language model.

## Notebook data flow

| Stage | File written | Read by |
| --- | --- | --- |
| Preprocessing | `dataset_preprocessed.csv` | NLP enrichment |
| NLP enrichment | `dataset_enriched.pkl` | Exploration and mood assignment |
| Basic mood assignment | `dataset_mood_adjusted.pkl` and `dataset_mood_adjusted.csv` | Derived-label F1 calculation and complex mood assignment |
| Derived-label preparation | `dataset_with_ground_truth.pkl` | F1 calculation |
| Complex mood assignment | `dataset_with_complex_moods_fixed.csv` | Final notebook output |

The pickle retains NumPy embedding arrays. The CSV carries those arrays as text for the later complex mood cell, which parses them back into vectors. Generated files are local outputs, not source data.

The notebook calls labels derived from `mood_composite_str` “ground truth” and compares a prediction derived from the same mood assignments against them. Its F1 values therefore measure agreement with its own labels; they do not establish accuracy against independent human annotations or support comparisons with other models. The notebook also does not test whether users find its movie results relevant.

To check the mood-data handoff without downloading models, run `python tests/check_notebook_dataflow.py` after installing the requirements.

## License

See [LICENSE](LICENSE).

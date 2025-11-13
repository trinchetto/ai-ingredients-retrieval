# ai-ingredients-retrieval

![Test Coverage](https://codecov.io/gh/trinchetto/ai-ingredients-retrieval/branch/main/graph/badge.svg)

Skeleton utilities for normalizing noisy ingredient strings with a locally running
[Ollama](https://ollama.com) model (default: `phi3:mini`). The project is packaged with
Poetry so dependencies and runnable entry points are easy to manage.

## Installation

```bash
poetry install
```

Poetry will create (or reuse) a virtual environment for the package. Make sure the
[Ollama](https://ollama.com) server is running locally (defaults to
`http://localhost:11434`) because the normalizer calls the REST API endpoint.

## Usage

The `ai_ingredients.ollama_normalizer` module exposes two main helpers:

- `preclean(text: str) -> str` trims, lowercases, and removes diacritics before sending
  text to the LLM.
- `normalize_with_llm(text: str, model: str = "phi3:mini") -> str` builds the few-shot
  prompt shown in `src/ai_ingredients/ollama_normalizer.py`, calls the Ollama REST API,
  and returns the last assistant message rendered as a normalized ingredient.

You can run the sample driver (also installed as the `normalize-ingredients` script) via:

```bash
poetry run python -m ai_ingredients.ollama_normalizer
# or
poetry run normalize-ingredients
```

Feel free to import `ai_ingredients.normalize_with_llm` inside other projects to reuse
the skeleton prompt/logic.

## Dataset normalization script

Use the dataset driver to read `dataset/Food Ingredients and Recipe Dataset with Image Name Mapping.csv`,
extract the `Cleaned_Ingredients` column, and write a JSON file containing normalized
ingredients generated through the Ollama helper:

```bash
poetry run python -m ai_ingredients.dataset_normalizer \
  --dataset "dataset/Food Ingredients and Recipe Dataset with Image Name Mapping.csv" \
  --output dataset/normalized_ingredients.json
# or
poetry run normalize-ingredients-dataset
```

Additional flags let you pick an alternate column, Ollama model, temperature,
`num-predict`, API URL, or request timeout. The output JSON mirrors the input structure
but with each entry normalized by the LLM.

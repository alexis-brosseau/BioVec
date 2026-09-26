# VecFuzz-derived DNA vector benchmarks

A small standalone benchmark repository for a handcrafted DNA representation derived from the vectorization code in [VecFuzz](https://github.com/alexis-brosseau/VecFuzz). The benchmark code does **not** depend on VecFuzz or FAISS: it uses the included `vectorizer.py`, NumPy/SciPy, and exact Canberra distance.

Current configuration:

```python
def make_vectorizer(chars: str) -> SequenceVectorizer:
    vectorizers = [
        Vectorizer.rbf.params(centers=[0.0, 1.0], sigma=0.0075),
        Vectorizer.bigram.params(dim=len(chars) ** 2),
    ]
    return SequenceVectorizer(chars=chars, vectorizers=vectorizers)
```

The alphabet is detected from each dataset at runtime and Canberra distance is used throughout.

## Install

```bash
python -m pip install -r requirements.txt
```

## AFproject CRM

Download the official [AFproject CRM dataset](https://afproject.org/app/benchmark/genreg/crm/dataset/) and the CRM scoring reference (`ids.json`) from the official AFproject source repository:

```bash
python crm/download.py
```

Generate all-versus-all Canberra distances:

```bash
python crm/run.py
```

This writes:

```text
results/crm_distances.tsv
```

The file is a headerless, three-column TSV:

```text
sequence_id_1<TAB>sequence_id_2<TAB>distance
```

Score the generated file locally with the same CRM ranking logic used by AFproject:

```bash
python crm/scorer.py
```

By default the scorer reads `results/crm_distances.tsv` and `datasets/crm/ids.json`, prints the per-tissue scores plus weighted average, standard deviation, and average, and saves the report to:

```text
results/crm_score.txt
```


## Genomic Benchmarks

Download the four datasets:

```bash
python genomic/download.py
```

Run exact 1-NN classification with Canberra distance:

```bash
python genomic/run.py
```

This writes the complete metrics and confusion matrices to:

```text
results/genomic_results.json
```

The included datasets are:

- `drosophila_enhancers_stark`
- `human_enhancers_cohn`
- `human_enhancers_ensembl`
- `human_ensembl_regulatory`

The data downloader uses the official [`genomic-benchmarks`](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) Python package.


## Recorded comparison

These are the experimental results used when preparing this repository. Both representations use **Canberra distance**. The runners themselves execute only the representation currently written in `config.py`.

### AFproject CRM

You can compare with the official [`leaderboard`](https://afproject.org/app/benchmark/genreg/crm/results/).

| Representation | Dimensions | Weighted score | Stdev |
|---|---:|---:|---:|
| Bigram only | 25 | 73.60% | 10.54 |
| Bigram + RBF (`sigma=0.0075`) | 35 | **77.03%** | 12.03 |

The bigram-only result reproduces the `alfpy--canberra` CRM result used as the comparison baseline. The 35-dimensional representation adds 10 localized endpoint-density features to the 25 DNA bigram dimensions.

### Genomic Benchmarks

Each cell is `1-NN accuracy / macro-F1`. You can compare with the official experimental runs of different CNN models [`genomic-benchmarks-experiments`](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/experiments/README.md).

| Dataset | Bigram only | Bigram + RBF |
|---|---:|---:|
| Drosophila enhancers | 56.71 / 56.71 | **65.72 / 65.71** |
| Human enhancers Cohn | **63.31 / 63.31** | 59.93 / 59.92 |
| Human enhancers Ensembl | 78.53 / 78.36 | **79.00 / 78.80** |
| Human Ensembl regulatory | 54.16 / 53.62 | **55.08 / 54.13** |


## Reproducing the bigram-only baseline

The repository deliberately has no benchmark mode selector. To rerun the bigram-only ablation, change the one configuration in `config.py` to:

```python
def make_vectorizer(chars: str) -> SequenceVectorizer:
    return SequenceVectorizer(
        chars=chars,
        vectorizers=[
            Vectorizer.bigram.params(dim=len(chars) ** 2),
        ],
    )
```
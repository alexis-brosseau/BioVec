from vectorizer import SequenceVectorizer, Vectorizer

# One frozen benchmark configuration. The alphabet is discovered from each
# dataset at runtime and passed into make_vectorizer().
DISTANCE = "canberra"

# Genomic Benchmarks evaluated in the README.
GENOMIC_DATASETS = (
    "drosophila_enhancers_stark",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
)

VECTORIZE_BATCH_SIZE = 2048
SEARCH_BATCH_SIZE = 128


def make_vectorizer(chars: str) -> SequenceVectorizer:
    vectorizers = [
        # Vectorizer.rbf.params(centers=[0.0, 1.0], sigma=0.0075),
        Vectorizer.ngram.params(n=2, dim=len(chars) ** 2),
    ]
    return SequenceVectorizer(chars=chars, vectorizers=vectorizers)

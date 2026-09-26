from typing import Callable
from dataclasses import dataclass
from functools import partial
import numpy as np
from math import comb


@dataclass
class VecContext:
    """
    A neutral, pre-computed representation of words.
    Contains no vector-specific logic; purely represents the parsed characters.
    """
    # --- Core Metadata ---
    words: list[str]                  # The original, cleaned strings
    num_chars: int                    # Total size of the character vocabulary
    
    # --- 1D Word-Level Properties ---
    word_lengths: np.ndarray          # Shape: (batch_size,). The actual length of each word.
    
    # --- 2D Grid (For sequence-aware operations like bigrams) ---
    char_matrix: np.ndarray           # Shape: (batch_size, max_len). Char IDs, -1 for pad/unknown.
    
    # --- 1D Flattened Arrays (For fast np.add.at on valid characters only) ---
    word_ids: np.ndarray              # Shape: (num_valid_chars,). Which word each valid char belongs to.
    char_ids: np.ndarray              # Shape: (num_valid_chars,). The vocabulary ID of each valid char.
    char_positions: np.ndarray        # Shape: (num_valid_chars,). 0-based position of the char in its word.
    expanded_word_lengths: np.ndarray # Shape: (num_valid_chars,). The word length, repeated for each valid char.
    flat_char_indices: np.ndarray     # Shape: (num_valid_chars,). Pre-computed (word_id * num_chars + char_id).


class VectorizerOp:
    """
    A callable wrapper that allows vectorizer functions to be multiplied 
    or divided by a scalar weight (e.g., `vectorizer * 2.0` or `vectorizer / 3.0`).
    """
    
    def __init__(self, func):
        self._func = func
        self.__name__ = getattr(func, '__name__', 'vectorizer')
        self.__doc__ = getattr(func, '__doc__', '')

    def __call__(self, ctx, *args, **kwargs):
        return self._func(ctx, *args, **kwargs)

    def __mul__(self, scalar):
        """Handles `vectorizer * scalar`"""
        def scaled(ctx, *args, **kwargs):
            return self._func(ctx, *args, **kwargs) * scalar
        op = VectorizerOp(scaled)
        op.__name__ = f"{self.__name__}_x{scalar}"
        return op

    def __rmul__(self, scalar):
        """Handles `scalar * vectorizer`"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """Handles `vectorizer / scalar`"""
        if scalar == 0: raise ZeroDivisionError("Cannot divide vectorizer by zero")
        def scaled(ctx, *args, **kwargs):
            return self._func(ctx, *args, **kwargs) / scalar
        op = VectorizerOp(scaled)
        op.__name__ = f"{self.__name__}_/{scalar}"
        return op

    def __rtruediv__(self, scalar):
        """Handles `scalar / vectorizer`"""
        if scalar == 0:
            def zero_func(ctx, *args, **kwargs):
                return np.zeros_like(self._func(ctx, *args, **kwargs))
            return VectorizerOp(zero_func)
        def scaled(ctx, *args, **kwargs):
            return scalar / self._func(ctx, *args, **kwargs)
        op = VectorizerOp(scaled)
        op.__name__ = f"{scalar}/{self.__name__}"
        return op

    def __pow__(self, exponent):
        """Handles `vectorizer ** exponent`"""
        def powered(ctx, *args, **kwargs):
            # Use np.power to handle negative bases with fractional exponents safely if needed,
            # but standard ** is fine for positive vectors.
            return np.power(self._func(ctx, *args, **kwargs), exponent)
        
        op = VectorizerOp(powered)
        op.__name__ = f"{self.__name__}^{exponent}"
        return op
    
    def __rpow__(self, base):
        """Handles `base ** vectorizer`"""
        def powered(ctx, *args, **kwargs):
            return np.power(base, self._func(ctx, *args, **kwargs))
        
        op = VectorizerOp(powered)
        op.__name__ = f"{base}^{self.__name__}"
        return op
    
    def __add__(self, other):
        """Handles `vectorizer + vectorizer` or `vectorizer + scalar`"""
        
        if isinstance(other, VectorizerOp):
            def added(ctx, *args, **kwargs):
                return self._func(ctx, *args, **kwargs) + other._func(ctx, *args, **kwargs)
            op = VectorizerOp(added)
            op.__name__ = f"{self.__name__}_plus_{other.__name__}"
            return op
        else:
            def added_scalar(ctx, *args, **kwargs):
                return self._func(ctx, *args, **kwargs) + other
            op = VectorizerOp(added_scalar)
            op.__name__ = f"{self.__name__}_plus_{other}"
            return op
        
    def __radd__(self, other):
        """Handles `scalar + vectorizer`"""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Handles `vectorizer - vectorizer` or `vectorizer - scalar`"""
        
        if isinstance(other, VectorizerOp):
            def subtracted(ctx, *args, **kwargs):
                return self._func(ctx, *args, **kwargs) - other._func(ctx, *args, **kwargs)
            op = VectorizerOp(subtracted)
            op.__name__ = f"{self.__name__}_minus_{other.__name__}"
            return op
        else:
            def subtracted_scalar(ctx, *args, **kwargs):
                return self._func(ctx, *args, **kwargs) - other
            op = VectorizerOp(subtracted_scalar)
            op.__name__ = f"{self.__name__}_minus_{other}"
            return op
        
    def __rsub__(self, other):
        """Handles `scalar - vectorizer`"""
        def subtracted_scalar(ctx, *args, **kwargs):
            return other - self._func(ctx, *args, **kwargs)
        op = VectorizerOp(subtracted_scalar)
        op.__name__ = f"{other}_minus_{self.__name__}"
        return op
    
    def norm(self, ord: int) -> "VectorizerOp":
        """
        Returns a new VectorizerOp that normalizes the output vector 
        to have a unit norm (L2 by default) per row (per word).
        
        Args:
            ord (int): The order of the norm. 2 for Euclidean (L2), 1 for Manhattan (L1).
        """
        def normed(ctx, *args, **kwargs):
            vec = self._func(ctx, *args, **kwargs)
            
            # Calculate norm per row (each word's vector gets normalized independently)
            norms = np.linalg.norm(vec, ord=ord, axis=1, keepdims=True)
            
            # Prevent division by zero for empty words or zero-vectors
            norms = np.where(norms == 0, 1.0, norms)
            
            return vec / norms
        
        op = VectorizerOp(normed)
        op.__name__ = f"norm{ord}({self.__name__})"
        return op
    
    def params(self, *args, **kwargs) -> "VectorizerOp":
        """
        EXPLICIT CONFIGURATION MODE.
        Binds parameters to this vectorizer and returns a new VectorizerOp.
        """
        # Using functools.partial is highly optimized in C
        bound_func = partial(self._func, *args, **kwargs)
        
        new_op = VectorizerOp(bound_func)
        new_op.__doc__ = self.__doc__
        
        # Generate a clean, readable name for debugging/introspection
        param_strs = [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
        if param_strs:
            new_op.__name__ = f"{self.__name__}({', '.join(param_strs)})"
        else:
            new_op.__name__ = self.__name__
            
        return new_op
    
    
def vectorizer(func):
    """Decorator to convert a raw vectorizer function into a VectorizerOp."""
    return VectorizerOp(func)


class Vectorizer:
    
    @staticmethod
    @vectorizer
    def length(ctx: VecContext) -> np.ndarray:
        """Returns a vector of the lengths of each word."""
        return ctx.word_lengths.astype(np.float32).reshape(-1, 1)
    
    @staticmethod
    @vectorizer
    def bernstein(
        ctx: VecContext,
        degree: int = 2,
        indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Character-position encoding using Bernstein basis polynomials.

        B_{k,n}(x) = C(n, k) * x^k * (1 - x)^(n-k)

        Args:
            degree:
                Bernstein polynomial degree n.

            indices:
                Which basis polynomials k to include.
                None means all: (0, 1, ..., degree).
        """
        if degree < 0:
            raise ValueError("degree must be >= 0")

        if indices is None:
            indices = tuple(range(degree + 1))

        if not indices:
            raise ValueError("indices cannot be empty")

        if len(set(indices)) != len(indices):
            raise ValueError("indices must be unique")

        if any(k < 0 or k > degree for k in indices):
            raise ValueError(
                f"indices must be between 0 and degree ({degree})"
            )

        n_words = ctx.word_lengths.shape[0]
        n_basis = len(indices)

        vec = np.zeros(
            (n_words, ctx.num_chars * n_basis),
            dtype=np.float32,
        )

        if n_words == 0:
            return vec

        # Position represented by the center of each character cell.
        x = (
            ctx.char_positions.astype(np.float32) + 0.5
        ) / ctx.expanded_word_lengths

        width = ctx.num_chars * n_basis
        flat = vec.reshape(-1)

        for basis_idx, k in enumerate(indices):
            weight = (
                comb(degree, k)
                * np.power(x, k)
                * np.power(1.0 - x, degree - k)
            ).astype(np.float32)

            target = (
                ctx.word_ids * width
                + basis_idx * ctx.num_chars
                + ctx.char_ids
            )

            np.add.at(flat, target, weight)

        return vec

    @staticmethod
    @vectorizer
    def phase(ctx: VecContext, freqs: list[float] = [0.75, 2.0]) -> np.ndarray:
        """
        Phase-encoded position: sinusoidal expansion of each character's position.
        
        Args:
            freqs (tuple[float]): Frequencies for the sinusoidal encoding. Duplicates are automatically weighted by sqrt(count).
        """
        n = ctx.word_lengths.shape[0]
        pos = (ctx.char_positions + 1) / ctx.expanded_word_lengths
        
        # 1. Count occurrences and extract unique frequencies (preserving original order)
        freq_counts = {}
        unique_freqs = []
        for f in freqs:
            if f not in freq_counts:
                freq_counts[f] = 0
                unique_freqs.append(f)
            freq_counts[f] += 1
            
        phase_parts = []
        for freq in unique_freqs:
            # 2. Calculate the L2-equivalent amplitude: sqrt(count)
            # e.g., if 1.0 appears twice, amplitude = sqrt(2) ≈ 1.414
            amplitude = np.sqrt(float(freq_counts[freq]))
            
            theta = freq * np.pi * pos
            cos_arr = np.zeros((n, ctx.num_chars), dtype=np.float32)
            sin_arr = np.zeros((n, ctx.num_chars), dtype=np.float32)
            
            # 3. Apply the amplitude scaling directly during the scatter operation
            np.add.at(cos_arr.reshape(-1), ctx.flat_char_indices, (amplitude * np.cos(theta) / ctx.expanded_word_lengths).astype(np.float32))
            np.add.at(sin_arr.reshape(-1), ctx.flat_char_indices, (amplitude * np.sin(theta) / ctx.expanded_word_lengths).astype(np.float32))
            
            phase_parts.extend([cos_arr, sin_arr])
            
        return np.concatenate(phase_parts, axis=1)
    
    @staticmethod
    @vectorizer
    def rbf(ctx: VecContext, centers: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0], sigma: float = 0.18) -> np.ndarray:
        """
        Radial Basis Function (Gaussian) positional encoding.
        
        It uses a Gaussian kernel, which is Smooth AND Local. 
        A deletion only shifts characters within the local "window" of the Gaussian. 
        Dimensions far from the deletion remain mathematically untouched, keeping the 
        L1 distance tiny while maintaining continuous positional awareness.
        
        Args:
            centers (tuple[float]): The relative positions (0.0 to 1.0) to place Gaussian centers.
            sigma (float): The width of the Gaussian. Controls how much overlap there is between centers.
        """
        n = ctx.word_lengths.shape[0]
        num_centers = len(centers)
        
        if n == 0:
            return np.zeros((0, ctx.num_chars * num_centers), dtype=np.float32)

        vec = np.zeros((n, ctx.num_chars * num_centers), dtype=np.float32)
        
        # Relative positions (0.0 to 1.0)
        rel_pos = ctx.char_positions.astype(np.float32) / ctx.expanded_word_lengths
        
        # Precompute the denominator for the Gaussian
        denom = 2.0 * (sigma ** 2)
        
        for c_idx, center in enumerate(centers):
            # Calculate Gaussian activation for this specific center
            # act shape: (num_valid_chars,)
            act = np.exp(-((rel_pos - center) ** 2) / denom)
            
            # Target flat index: word_id * (chars * centers) + char_id * centers + center_idx
            target = ctx.word_ids * (ctx.num_chars * num_centers) + ctx.char_ids * num_centers + c_idx
            
            # Accumulate
            np.add.at(vec.reshape(-1), target, act.astype(np.float32))
            
        # Normalize by word length so longer words don't dominate the magnitude
        vec /= ctx.word_lengths[:, None]
        
        return vec

    @staticmethod
    @vectorizer
    def ngram(
        ctx: VecContext,
        n: int = 2,
        dim: int = 192,
    ) -> np.ndarray:
        """
        Hashed n-gram frequency vectorization.

        Each contiguous n-gram is hashed into a fixed number of buckets.

        Args:
            n (int):
                Size of the n-gram. For example:
                1 = unigram
                2 = bigram
                3 = trigram

            dim (int):
                Number of hashing buckets in the output vector.
        """
        if n < 1:
            raise ValueError("ngram must be >= 1")

        n_words = ctx.word_lengths.shape[0]

        if n_words == 0:
            return np.zeros((0, dim), dtype=np.float32)

        max_len = ctx.char_matrix.shape[1]

        if max_len < n:
            return np.zeros((n_words, dim), dtype=np.float32)

        width = max_len - n + 1

        # Start with every possible n-gram window as valid.
        valid = np.ones((n_words, width), dtype=bool)

        # Polynomial rolling-style encoding of the character IDs.
        #
        # Since char IDs are in [0, num_chars), this produces a unique
        # integer representation before hashing as long as it fits.
        n_ids = np.zeros((n_words, width), dtype=np.int64)

        for offset in range(n):
            chars = ctx.char_matrix[:, offset : offset + width]

            valid &= chars >= 0

            n_ids *= ctx.num_chars
            n_ids += np.where(chars >= 0, chars, 0)

        word_ids, positions = np.nonzero(valid)

        if len(word_ids) == 0:
            return np.zeros((n_words, dim), dtype=np.float32)

        ids = n_ids[word_ids, positions]

        # Hash into fixed-dimensional buckets.
        buckets = ids % dim

        vec = np.zeros((n_words, dim), dtype=np.float32)

        flat_indices = word_ids * dim + buckets

        # Normalize by number of possible n-grams in the word rather than
        # raw word length.
        denominators = np.maximum(
            ctx.word_lengths[word_ids] - n + 1,
            1,
        ).astype(np.float32)

        np.add.at(
            vec.reshape(-1),
            flat_indices,
            1.0 / denominators,
        )

        return vec


def build_context(self, words: list[str]) -> VecContext:
    """Parses strings into the shared, neutral VecContext. Happens exactly once per batch."""
    words = [w.strip().lower() for w in words]
    n = len(words)
    
    if n == 0:
        return VecContext(
            words=[], num_chars=self._chars_len,
            word_lengths=np.array([], dtype=np.int64),
            char_matrix=np.empty((0, 0), dtype=np.int64),
            word_ids=np.array([], dtype=np.int64),
            char_ids=np.array([], dtype=np.int64),
            char_positions=np.array([], dtype=np.int64),
            expanded_word_lengths=np.array([], dtype=np.float32),
            flat_char_indices=np.array([], dtype=np.int64)
        )

    lengths = np.array([len(w) for w in words], dtype=np.int64)
    max_len = int(lengths.max())
    
    # 1. Build the neutral 2D char_matrix
    char_matrix = np.full((n, max_len), -1, dtype=np.int64)
    for r, w in enumerate(words):
        for c, ch in enumerate(w):
            char_matrix[r, c] = self._char_idx.get(ch, -1)
            
    # 2. Extract 1D flattened arrays for valid characters only
    i = np.arange(max_len)[None, :]
    mask = (i < lengths[:, None]) & (char_matrix >= 0)
    word_ids, char_positions = np.nonzero(mask)
    
    char_ids = char_matrix[word_ids, char_positions]
    exp_lengths = lengths[word_ids].astype(np.float32)
    flat_indices = word_ids * self._chars_len + char_ids

    return VecContext(
        words=words, 
        num_chars=self._chars_len,
        word_lengths=lengths,
        char_matrix=char_matrix,
        word_ids=word_ids, 
        char_ids=char_ids, 
        char_positions=char_positions,
        expanded_word_lengths=exp_lengths, 
        flat_char_indices=flat_indices
    )


class SequenceVectorizer:
    """Small standalone encoder built around the Vectorizer operations above."""

    def __init__(self, chars: str, vectorizers: list[VectorizerOp]):
        if not chars:
            raise ValueError("chars cannot be empty")
        if len(set(chars)) != len(chars):
            raise ValueError("chars must contain unique characters")
        if not vectorizers:
            raise ValueError("at least one vectorizer is required")

        self.chars = chars.lower()
        self._char_idx = {ch: i for i, ch in enumerate(self.chars)}
        self._chars_len = len(self.chars)
        self.vectorizers = list(vectorizers)

    def transform(self, words: list[str]) -> np.ndarray:
        ctx = build_context(self, words)
        if not words:
            return np.empty((0, 0), dtype=np.float32)

        parts = [np.asarray(op(ctx), dtype=np.float32) for op in self.vectorizers]
        if not parts:
            return np.empty((len(words), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    # Familiar alias for callers that prefer the original VecFuzz wording.
    vectorize = transform

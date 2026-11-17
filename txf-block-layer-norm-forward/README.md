# Layer Normalization, forward

Intermediate | transformers

### The problem, from first principles

`[03-dl-training]` showed why deep networks are hard to train: activations can drift to wildly different scales layer over layer, which destabilizes gradients and slows (or entirely stalls) learning. Batch Normalization was one classic fix, but it normalizes ACROSS the batch, which makes it awkward for sequence models: batches of sequences vary in length, and at generation time a model often processes ONE token at a time, where "the batch" barely exists as a meaningful statistical population at all.

Layer Normalization sidesteps this entirely by normalizing across the FEATURE axis instead, independently for each individual position (each token, each batch element) rather than across a batch. Every position's own `d_model`-length vector gets rescaled to zero mean and unit variance, using only that position's own values, then a learned per-feature scale (`gamma`) and shift (`beta`) restore whatever scale and offset the network actually needs, learned rather than fixed. Because the normalization no longer depends on other members of the batch, this works identically whether a model processes one token or a thousand at once, which is exactly why it, not Batch Normalization, became the standard choice for Transformers.

### From theory to code

Implement `layer_norm_forward(x, gamma, beta, eps)`. Compute the mean and variance along the LAST axis of `x` (the feature axis), normalize `x` using those per-position statistics, then apply the learned `gamma` (multiplicative) and `beta` (additive) parameters, both shaped like a single feature vector (broadcasting across every leading batch/sequence dimension).

### Constraints

- Statistics (mean, variance) are computed along the LAST axis only, independently for every other (batch, sequence) position, never across the batch or across positions.
- Use the BIASED variance (dividing by `n`, not `n - 1`): this is what `torch.nn.LayerNorm` uses internally, and using the Bessel-corrected sample variance instead gives a subtly wrong scale.
- `eps` is added INSIDE the square root, `sqrt(var + eps)`, purely to avoid dividing by zero when a position's variance is exactly (or very nearly) zero.
- `gamma`/`beta` are applied AFTER normalizing, as `gamma * x_norm + beta`, not before.

### Hints

<details>
<summary>Hint 1: The two statistics</summary>

`mean = x.mean(axis=-1, keepdims=True)`, `var = x.var(axis=-1, keepdims=True)`. NumPy's `.var()` already uses the biased (`ddof=0`) formula by default, exactly matching PyTorch's convention here, no extra argument needed.

</details>

<details>
<summary>Hint 2: Normalizing</summary>

`x_norm = (x - mean) / np.sqrt(var + eps)`. `keepdims=True` on both statistics keeps this a clean broadcast against `x`'s original shape, no manual reshaping required.

</details>

<details>
<summary>Hint 3: The learned affine transform</summary>

`return gamma * x_norm + beta`. `gamma`/`beta` are 1D, shaped `(d_model,)`; NumPy broadcasts them against `x_norm`'s last axis automatically, regardless of how many leading batch/sequence dimensions `x` has.

</details>

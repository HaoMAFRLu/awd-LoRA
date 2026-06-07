# Structured Input Embeddings as Semantic Bottlenecks

This note records the research idea and the concrete implementation plan for a follow-up project in this repository. The goal is not to make the embedding layer smaller for deployment. The goal is to study, theoretically and empirically, whether training-time sparse-plus-low-rank structure in the input embedding layer induces a better lexical representation and improves transfer/generalization.

## 1. Core Question

Standard language models map text into continuous space through two steps:

```text
raw text -> tokenizer -> token ids -> input embedding layer -> continuous token representations
```

The tokenizer defines a fixed discrete vocabulary. The input embedding matrix assigns each vocabulary item a continuous vector. In standard training, each token row in the embedding matrix has largely independent degrees of freedom. This can let the model encode useful lexical/semantic information, but it can also let the model memorize corpus-specific and token-specific artifacts such as frequency, BPE segmentation artifacts, rare-token noise, punctuation behavior, and domain-specific co-occurrence patterns.

The main hypothesis is:

> Training-time sparse-plus-low-rank induction in the input embedding layer acts as a semantic bottleneck. The low-rank component captures shared lexical-semantic structure, while the sparse component preserves necessary token-specific exceptions. This can improve downstream transfer and OOD generalization by suppressing token-specific empirical noise.

The project should explicitly avoid presenting this as an embedding compression paper. Parameter reduction is secondary and should not be the main contribution.

## 2. Theoretical Framing

The book `Principles and Practice of Deep Representation Learning` motivates the following logic:

1. Learning should recover low-dimensional structure in high-dimensional data, not memorize finite samples.
2. Good representations are lossy but useful codes: they should discard nuisance details while preserving task-relevant structure.
3. Overly flexible models can fit the empirical distribution and memorize sample-specific artifacts.
4. A structured bottleneck can improve generalization when the true signal has lower-dimensional shared structure and the remaining idiosyncratic part is sparse.

For token embeddings, the embedding matrix is special because it is both a parameter matrix and the representation table:

```math
z_i = E[token_i].
```

Thus, imposing structure on the input embedding matrix directly imposes structure on lexical representations. This is unlike imposing low-rank structure on a generic Transformer block, where parameter structure is not automatically representation structure.

Use the model:

```math
E^\star = L^\star + S^\star + N,
```

where:

- `L*` is a low-rank shared lexical-semantic component.
- `S*` is a sparse token-specific exception component.
- `N` is finite-sample/corpus-specific/token-frequency noise.

Downstream tasks are assumed to depend mostly on `L*` and on a small subset of meaningful exceptions in `S*`, not on arbitrary noise `N`.

Training a dense embedding can fit both signal and noise:

```math
\hat E_dense = L^\star + S^\star + N + estimation noise.
```

Training with sparse-plus-low-rank induction gives:

```math
\hat E_SLR = \hat L + \hat S.
```

The expected argument is a bias-variance tradeoff:

```math
excess risk <= approximation error + estimation error.
```

SLR may introduce some approximation bias but reduces estimation error, especially for rare tokens and domain-shifted downstream tasks. The theoretical result can be formulated under assumptions such as:

- shared lexical structure is approximately low-rank;
- token-specific exceptions are sparse;
- rare token embeddings have higher estimation noise;
- downstream labels depend primarily on the shared semantic structure;
- the sparse exception component is useful only for a small token subset.

The desired theorem/proposition does not need to prove that compression always helps. It should prove that under a reasonable latent semantic model, a structured estimator can have lower transfer/OOD risk than an unconstrained dense estimator.

## 3. Important Distinction: Input Embedding vs LM Head

The input embedding and LM head are trained together through the same next-token prediction loss, but they play different roles.

Input embedding:

```math
token id -> continuous representation.
```

It is an encoder-like lexical coding layer. It can be lossy as long as task-relevant semantic information is preserved.

LM head:

```math
hidden state -> next-token logits over vocabulary.
```

It is a decoder/classifier over the vocabulary. It must preserve fine-grained output discrimination. Forcing the LM head into the same sparse-low-rank structure may hurt because next-token prediction requires distinguishing many similar tokens and long-tail vocabulary items.

This leads to a key empirical prediction:

> Input embeddings should exhibit benign SLR-inducibility, while LM heads should resist the same structure or suffer more performance degradation.

This asymmetry is central. The project should include explicit input embedding vs LM head comparisons.

Also distinguish tied vs untied settings:

- In tied models, the input embedding and LM head share one matrix. This confounds the interpretation.
- In untied models, input embedding and LM head are separate matrices. This is the cleaner setup for the main experiments.

Main experiments should use untied embeddings if possible. Tied vs untied should be an ablation.

## 4. Experimental Goal

The goal is to build a full evidence chain:

```text
training-time SLR induction
-> changed embedding geometry
-> reduced token-frequency / rare-token noise
-> improved low-resource, OOD, or long-tail downstream generalization
-> effect is specific to input embedding and not reproduced by generic regularization
```

Do not rely only on validation perplexity or GLUE average. The most convincing result would be:

> Dense embedding may match or slightly outperform on training loss, but SLR input embedding transfers better under low-resource, OOD, domain-shift, or rare-token-heavy settings.

## 5. Model and Training Comparisons

Implement or configure the following variants. The names below should be used consistently in logs and result tables.

### 5.1 Main Variants

1. `dense`
   - Standard dense input embedding.
   - Standard dense LM head.

2. `slr_input_embedding`
   - Apply SALAAD/SLR induction only to the input embedding matrix.
   - Keep LM head dense.
   - This is the core method.

3. `slr_lm_head`
   - Keep input embedding dense.
   - Apply SLR only to the LM head.
   - This tests the asymmetry hypothesis.

4. `slr_input_and_lm_head`
   - Apply SLR to both input embedding and LM head.
   - Useful to show whether the LM head damages the benefit.

5. `slr_all_blocks_no_embedding`
   - Apply SLR to Transformer blocks only, excluding input embedding and LM head.
   - This separates embedding-specific effects from general model regularization.

6. `slr_all_blocks_with_embedding`
   - Full SALAAD-style model including input embedding, excluding LM head by default.

### 5.2 Baselines to Rule Out Trivial Explanations

7. `dense_small_embedding`
   - Dense embedding with comparable parameter count to SLR input embedding.
   - Rules out the explanation that "fewer parameters" alone helps.

8. `posthoc_svd_embedding`
   - Train dense model, then apply SVD/low-rank approximation to the input embedding.
   - Rules out post-hoc compression as sufficient.

9. `posthoc_rpca_embedding`
   - Train dense model, then apply RPCA or sparse-plus-low-rank decomposition.
   - Compare to training-time induction.

10. `low_rank_only_embedding`
    - Use only low-rank induction, no sparse component.
    - Tests whether sparse exceptions are needed.

11. `sparse_only_embedding`
    - Use only sparse induction, no low-rank component.
    - Tests whether shared low-rank structure is needed.

12. `dense_embedding_dropout`
    - Dense model with embedding dropout.
    - Tests generic regularization.

13. `dense_weight_decay_strong`
    - Dense model with stronger weight decay on embedding.
    - Tests generic regularization.

14. `dense_norm_penalty`
    - Dense model with row-norm or Frobenius-norm regularization on embedding.
    - Tests whether norm control alone explains the effect.

If compute is limited, prioritize:

```text
dense
slr_input_embedding
slr_lm_head
dense_small_embedding
posthoc_svd_embedding
low_rank_only_embedding
```

## 6. Model Scales and Seeds

Use at least two scales, preferably three:

- 60M
- 130M
- 350M

If resources allow, add 1B. The effect may be strongest in small or medium models where the embedding layer is a larger fraction of the model and finite-sample/rare-token noise is more visible.

Use at least 3 random seeds for the core variants:

```text
dense
slr_input_embedding
slr_lm_head
dense_small_embedding
```

For expensive variants, one seed is acceptable initially, but the final paper should report uncertainty for the main claims.

## 7. Pretraining-Level Evaluation

Track:

1. training loss;
2. validation perplexity or cross-entropy;
3. train-validation gap;
4. per-token negative log-likelihood grouped by token frequency;
5. rank/density evolution of input embedding;
6. rank/density evolution of LM head if SLR is applied there;
7. total parameter count only as secondary information.

Expected effects:

- `slr_input_embedding` should match dense validation PPL or improve it slightly.
- Training loss may be slightly worse than dense, but validation gap should shrink.
- `slr_lm_head` should hurt more than `slr_input_embedding`.
- `posthoc_svd_embedding` and `posthoc_rpca_embedding` should underperform training-time SLR.
- `low_rank_only_embedding` may hurt rare/special tokens because it lacks exception capacity.
- SLR with a sparse exception component should be more stable than low-rank-only.

## 8. Embedding Geometry and Lexical Analysis

These analyses are central because they test whether the embedding space actually changed in the predicted way.

Use the input embedding matrix after training. For SLR, analyze:

- full `E = L + S`;
- low-rank component `L`;
- sparse component `S`;
- dense baseline embedding.

### 8.1 Effective Rank / Spectral Entropy

Compute the singular values of the embedding matrix.

Let:

```math
p_i = sigma_i^2 / sum_j sigma_j^2
```

and:

```math
r_eff = exp(-sum_i p_i log p_i).
```

Interpretation:

- Lower effective rank in `L` suggests shared low-dimensional lexical structure.
- The full `L + S` should not collapse too much.
- A long noisy singular-value tail in dense embeddings may indicate token-specific artifacts.

Expected:

- `L` has clearly lower effective rank.
- `L + S` preserves enough capacity.
- Better downstream generalization correlates with reduced noisy spectral tail, not with collapse.

### 8.2 Isotropy

Measure whether the embedding space is dominated by a few common directions.

Possible metrics:

- top principal component variance ratio;
- average pairwise cosine similarity;
- IsoScore if available;
- anisotropy before and after removing the mean.

Important nuance:

SLR reduces rank, so do not simply expect full-space isotropy to increase. Instead, evaluate whether the effective subspace is less dominated by frequency/punctuation/common-token directions.

Expected:

- Dense embeddings may have top PCs correlated with frequency, punctuation, or special tokens.
- SLR should reduce meaningless common directions.
- In the active low-rank subspace, variance should be more semantically organized.

### 8.3 Hubness

Hubness measures whether a small number of tokens become nearest neighbors of many other tokens.

Procedure:

1. Normalize embedding rows.
2. For each token, find top-k nearest neighbors by cosine similarity.
3. Count how often each token appears in other tokens' top-k lists.
4. Report skewness, Gini coefficient, and top hub tokens.

Expected:

- Dense embeddings may have hubs that are punctuation, high-frequency tokens, whitespace tokens, or common BPE fragments.
- SLR should reduce pathological hubs.
- Pure low-rank may increase hubness due to over-smoothing; SLR should be better than low-rank-only.

### 8.4 Token Norm vs Frequency

Compute each token row norm:

```math
||e_i||_2
```

and correlate it with log token frequency:

```math
corr(||e_i||_2, log freq_i).
```

Use Spearman and Pearson correlation.

Expected:

- Dense embeddings may encode frequency strongly in row norm.
- SLR should reduce or smooth this dependency.
- The sparse component may retain frequency-sensitive or exception-sensitive information for a small token subset.

Do not require zero correlation. Some frequency information is useful. The expected result is reduced pathological frequency dependence.

### 8.5 Nearest-Neighbor Semantic Purity

Evaluate whether embedding nearest neighbors are semantically or lexically meaningful.

Because BPE tokens can be noisy, start with a filtered token subset:

- alphabetic tokens only;
- length >= 3;
- exclude special tokens;
- optionally exclude tokens with leading whitespace marker depending on tokenizer.

Possible purity labels:

- same stem or lemma;
- same lowercase form;
- same prefix/suffix family;
- WordNet relation if applicable;
- POS tag;
- entity category;
- external embedding/LLM-assisted cluster label;
- context-distribution cluster.

Metrics:

- precision@k of same semantic/lexical group;
- nearest-neighbor overlap with external semantic embedding;
- qualitative nearest-neighbor tables for representative tokens.

Expected:

- SLR should improve semantic purity, especially for medium/rare tokens.
- `L` should show broad semantic neighborhoods.
- `S` should mainly modify exceptions rather than destroy neighborhoods.

### 8.6 Rare Token Embedding Stability

Rare tokens receive fewer gradient updates and should have noisier dense embeddings.

Procedure:

1. Train multiple seeds or multiple data-order variants.
2. Align embedding spaces using orthogonal Procrustes alignment on a stable frequent-token subset.
3. For each token, compute cross-seed cosine similarity of its embedding.
4. Group tokens by frequency bin.

Frequency bins can be:

- very rare;
- rare;
- medium;
- frequent;
- very frequent.

Expected:

- Dense rare-token embeddings are less stable across seeds.
- SLR rare-token embeddings are more stable.
- Frequent token stability should be similar across methods.
- Stability should not come from collapse; verify with semantic purity and per-token NLL.

### 8.7 Sparse Component Analysis

For SLR, analyze the sparse component `S`.

For each token row:

```math
||S_i||_0, ||S_i||_1, ||S_i||_2, ||S_i||_1 / ||E_i||_1
```

Group tokens by:

- frequency;
- special tokens;
- punctuation;
- whitespace tokens;
- numbers;
- capitalization;
- proper nouns;
- domain-specific terms;
- rare tokens;
- common subwords.

Expected:

- Sparse mass should not be uniformly distributed over all tokens.
- `S` should concentrate on tokens that plausibly need exceptions.
- Ordinary semantic tokens should rely more on `L`.
- Special tokens, numbers, punctuation, rare/domain tokens may need more sparse correction.

This analysis is important for interpreting `S` as meaningful exceptions rather than arbitrary compensation for low-rank approximation.

## 9. Downstream Generalization Evaluation

This is the most important empirical claim. The prediction is not necessarily that SLR wins on every full-data benchmark. The prediction is that SLR input embedding improves settings where lexical generalization matters.

### 9.1 Low-Resource Fine-Tuning

Evaluate with limited labeled data:

- 16 examples per class;
- 32 examples per class;
- 128 examples per class;
- 512 examples per class.

Tasks:

- GLUE subsets if using encoder-style adaptation is possible;
- otherwise causal LM-compatible classification via prompting or lightweight classification head;
- sentiment, topic, NLI, paraphrase tasks.

Expected:

- Dense and SLR may be similar with full data.
- SLR should be better and less variable in low-resource settings.

### 9.2 OOD / Domain Shift

Train/fine-tune on one domain and evaluate on shifted domains.

Possible evaluations:

- in-domain OpenWebText validation vs out-of-domain validation corpora;
- news -> biomedical;
- general web -> legal;
- general web -> code/docstrings;
- MNLI -> HANS if using NLI setup.

Expected:

- SLR should have smaller degradation from in-domain to OOD.
- Gains should be stronger in domains with rare or shifted vocabulary.

### 9.3 Long-Tail / Rare-Token Evaluation

Evaluate language modeling NLL by token frequency bins.

Procedure:

1. Compute token frequency on pretraining corpus.
2. Group validation tokens by frequency bin.
3. Report average NLL per bin.

Expected:

- Dense may do well on very frequent tokens.
- SLR should improve or stabilize rare/medium tokens.
- Low-rank-only may over-smooth rare tokens.
- SLR should outperform low-rank-only because sparse exceptions preserve necessary idiosyncrasies.

### 9.4 Retrieval / Semantic Similarity

If feasible, evaluate sentence or word-level embedding quality. This may require using the model as a feature extractor.

Potential tasks:

- MTEB subsets;
- word similarity benchmarks;
- lexical substitution or analogy-style probes;
- nearest-neighbor semantic purity as a lighter alternative.

Expected:

- SLR should improve semantic organization, but these results may be secondary because causal LMs are not optimized as sentence embedding models.

## 10. Key Figures and Tables

The final write-up should include:

1. Input embedding vs LM head SLR performance table.
2. Validation PPL vs variant table.
3. Low-resource downstream score vs number of fine-tuning samples.
4. OOD score or OOD PPL degradation plot.
5. Token frequency bin vs NLL plot.
6. Singular-value spectra for dense, `L`, `L + S`.
7. Effective rank / spectral entropy table.
8. Token frequency vs embedding norm scatter plot.
9. Hubness top-token table and Gini/skewness plot.
10. Sparse component mass by token category/frequency.
11. Rare-token cross-seed stability plot.
12. Qualitative nearest-neighbor examples for dense vs SLR.

## 11. Expected Result Pattern

The ideal result is:

```text
Dense embedding:
  lower or similar training loss,
  stronger token-frequency artifacts,
  more rare-token instability,
  worse low-resource/OOD/long-tail transfer.

SLR input embedding:
  similar validation PPL,
  smaller train-validation gap,
  cleaner spectral structure,
  reduced hubness/frequency bias,
  more stable rare-token embeddings,
  better low-resource/OOD/long-tail downstream performance.

SLR LM head:
  more likely to hurt PPL and downstream performance,
  supporting input/output asymmetry.

Post-hoc SVD/RPCA:
  does not reproduce training-time SLR benefits,
  supporting the importance of induction during learning.
```

This pattern would support the central claim that the structured embedding is a better semantic coding layer, not merely a smaller matrix.

## 12. Failure Modes and Diagnostics

### Failure Mode 1: SLR improves compression but not generalization

Diagnosis:

- The method may be too weak or too strong.
- Downstream tasks may not be sensitive to lexical generalization.
- Need low-resource/OOD/rare-token tasks.

### Failure Mode 2: SLR hurts PPL significantly

Diagnosis:

- Rank too low.
- Sparse budget too small.
- Induction starts too early.
- LM head accidentally tied or constrained.
- Embedding and hidden dimension mismatch.

### Failure Mode 3: Low-rank-only matches SLR

Diagnosis:

- Sparse exceptions may not matter for the chosen tokenizer/model/data.
- Need rare-token-heavy or domain-specific evaluation.
- Analyze whether `S` is actually used.

### Failure Mode 4: Dense-small matches SLR

Diagnosis:

- Effect may be generic capacity reduction, not structured semantic induction.
- Need geometry and `S` analysis.
- Need compare at multiple scales and data sizes.

### Failure Mode 5: SLR gains vanish in tied embedding models

Diagnosis:

- Tying mixes input representation and output classification roles.
- This supports the asymmetry hypothesis rather than refuting it.

## 13. Implementation Notes for This Repository

Likely relevant files/directories:

- `models/`: model definitions, including Llama/nanoGPT variants.
- `salad/`: SLR/SALAAD modules and training logic.
- `scripts/`: training/evaluation scripts and config generation.
- `analysis/embedding_comparison.py`: likely a good starting point for embedding-specific analysis.
- `test/plot_lm_head.py`: may contain useful LM head analysis logic.

Suggested implementation steps:

1. Add config flags:
   - `slr_target=input_embedding`
   - `slr_target=lm_head`
   - `slr_target=input_embedding,lm_head`
   - `slr_target=all_except_lm_head`
   - `tie_word_embeddings=true/false`

2. Ensure the input embedding module and LM head can be registered separately in SALAAD.

3. Add explicit logging of:
   - input embedding rank/density;
   - LM head rank/density;
   - `L`, `S`, and `L+S` statistics;
   - token frequency bins.

4. Add analysis scripts:
   - `analysis/embedding_spectrum.py`
   - `analysis/embedding_geometry.py`
   - `analysis/token_frequency_analysis.py`
   - `analysis/nearest_neighbor_purity.py`
   - `analysis/sparse_component_analysis.py`
   - `analysis/rare_token_stability.py`

5. Add downstream or proxy evaluation scripts:
   - per-token-frequency NLL;
   - OOD validation PPL;
   - low-resource fine-tuning if feasible;
   - prompt-based classification if full fine-tuning is too expensive.

6. Keep all analysis outputs reproducible:
   - save token frequencies;
   - save tokenizer metadata;
   - save config YAMLs;
   - save model variant names consistently;
   - save random seeds.

## 14. Claim Boundaries

Be careful not to overclaim.

Do not claim:

- SLR always improves embeddings.
- Compressed embeddings are universally better.
- The book directly proves embedding weights should be SLR.
- Lower rank itself is the goal.

Claim instead:

- Input embeddings are a special parameter matrix whose rows are token representations.
- Training-time SLR is a controlled structural intervention.
- Under a shared-semantic-plus-sparse-exception model, such a bottleneck can reduce estimation noise and improve transfer.
- Empirically, this should appear as better low-resource/OOD/long-tail generalization and cleaner lexical geometry.

## 15. Possible Paper Positioning

Potential titles:

- `Input Embeddings as Semantic Bottlenecks`
- `Why Input Embeddings Want Structure`
- `Structured Lexical Representations Improve Transfer Generalization`
- `On Sparse-Low-Rank Structure in Language Model Input Embeddings`

Potential contribution statement:

1. We identify and formalize a structural asymmetry between input embeddings and LM heads.
2. We propose a semantic bottleneck view of input embeddings based on shared low-rank lexical structure plus sparse token-specific exceptions.
3. We use training-time SLR induction as a controlled intervention to test this view.
4. We show that SLR input embeddings improve low-resource, OOD, or long-tail generalization while post-hoc compression and generic regularization do not fully reproduce the effect.
5. We provide embedding geometry and sparse-component analyses supporting the proposed mechanism.


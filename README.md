# rashomon-align

Measure how similar two models actually are — not just where your test data happens to live.

An implementation of **Rashomon Alignment** ([arXiv:2607.25680](https://arxiv.org/abs/2607.25680),
Santos, van der Putten, Pfahringer & Soares, July 2026), with a partial reproduction of the
paper's tabular results.

## The problem

Two models can score identically on your test set and still define completely different decision
boundaries everywhere else. Standard similarity measures compare predictions on observed data,
so they are only ecologically valid in the regions your data covers. Swap, prune, quantise or
fine-tune a model, hit a covariate shift, and the divergence you never measured becomes the
failure you did not predict.

The paper separates two questions:

| | Reference distribution | Answers |
| --- | --- | --- |
| **dRA** distributional | `P(X)` — your data | Do these models agree where my data lives? |
| **gRA** geometric | `U(F)` — uniform over the instance space | Do these models agree *anywhere else*? |

They are complementary, not competing. The paper's finding is that dRA concentrates *above* gRA,
so a high dRA can overestimate how structurally similar two models really are.

## Usage

```python
from rashomon_align import InstanceSpace, dra, gra

space = InstanceSpace.from_data(X_train)      # bounding box of the training features

print(dra(model_a, model_b, X_test).value)    # agreement on observed data
print(gra(model_a, model_b, space).value)     # agreement across the instance space
```

Anything with a `.predict` method works, as does any plain callable. The agreement function is
pluggable, so the measure extends past single-label classification:

```python
gra(model_a, model_b, space, agree=lambda a, b: mean_field_match(a, b))
```

## Reproduction

`experiments/reproduce_paper.py` follows the paper's protocol — unpruned versus cost-complexity
pruned decision trees, 5-fold stratified cross-validation, seed 42, 1000 uniform samples per fold
— across 29 OpenML datasets.

| Measure | This implementation | Paper |
| --- | --- | --- |
| mean dRA | 0.682 | — |
| mean gRA | 0.579 | — |
| dRA exceeds gRA | **22 of 29 datasets** | reported as the general pattern |
| `r(gRA, dRA)` | **0.725** | 0.745 |
| `r(gRA, Δaccuracy)` | **0.147** | 0.514 |

**What replicates.** The central claim holds clearly: models agree more on observed data than
across the instance space, in 22 of 29 datasets. The correlation between the two measures comes
out at 0.725 against the paper's 0.745.

**What does not.** The correlation between geometric alignment and the accuracy cost of pruning
is far weaker here — 0.147 against 0.514. The sign is right, the strength is not. The most likely
cause is pruning severity: taking the largest α from the cost-complexity path produces very
aggressive pruning, costing 11 accuracy points on average and up to 68 on one dataset. Under that
much pruning the accuracy penalty is dominated by how prunable a dataset is, which is not what
gRA measures. Dataset selection differs too — 29 OpenML datasets against the paper's 92 UCI ones.

This is reported as a partial replication rather than a clean one.

### Two things the reproduction caught

Both were bugs in this implementation, found by taking a mismatch seriously instead of adjusting
the number:

- The first run gave `r(gRA, Δaccuracy) = −0.563` against the paper's `+0.514`. The paper
  correlates gRA with **signed** accuracy difference (pruned minus unpruned); using the absolute
  value inverts the relationship.
- Pruning strength is selected **per fold** from the cost-complexity path, not fixed in advance.
  Hardcoding `ccp_alpha` changed `r(gRA, dRA)` from 0.859 to 0.725 — away from a flattering number
  and toward the paper's.

One genuine ambiguity remains. Taking the literal largest α collapses the tree to a single node;
scikit-learn's own documentation discards that trivial case. `MaximallyPrunedTree` falls back to
the second-largest α when the tree degenerates, which is a choice the paper does not specify.

## Applied to a text extraction model

The paper is tabular — `gRA` needs a bounded feature space to sample uniformly. Sequence models
over natural language have no such box.

[kid-extract](https://github.com/Chenjigaram/kid-extract-lora) is a case where they do, because
its documents are rendered from a bounded configuration space: layout, language, label wording,
field presence, value ranges. Sampling that space uniformly gives a genuine `U(F)`, including
documents no real provider would ever publish.

Comparing two rule-based extractors there — one knowing the training vocabulary, one knowing all
of it:

| | Strict agreement |
| --- | --- |
| dRA, realistic documents | **1.000** |
| gRA, uniform over configuration space | **0.207** |

Distributional alignment calls them the same model. They disagree on four documents in five once
you leave the data manifold. The disagreements fall entirely on label-dependent fields; identifier,
title and numeric-table fields show none, because those are recoverable by shape rather than by
name — a partition recovered here **without any ground truth at all**.

## Install

```bash
pip install -e ".[tabular,dev]"
pytest -q
python experiments/reproduce_paper.py
```

## Licence

MIT. The paper is the authors' work; this repository is an independent implementation.

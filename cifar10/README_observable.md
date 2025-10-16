# Observable-Based Influence for CIFAR-10

This implementation extends Kronfluence to compute **observable-based influence** scores instead of standard loss-based influence.

## Mathematical Framework

### Observable
Instead of optimizing for loss reduction, we optimize for increasing a specific probability:

```
f(θ) = log p(y*|x*; θ)
```

Where:
- `x*` is a query image
- `y*` is a target class (different from predicted class)
- `θ` are model parameters

### Inverse Hessian Vector Product (IHVP)

We compute:
```
v = (H + λI)^{-1} ∇_θ f
```

Where:
- `H` is the Gauss-Newton Hessian approximation via EK-FAC
- `λ` is the damping factor
- `∇_θ f` is the gradient of the observable (not the loss!)

### Influence Scores

For each training example `(x_i, y_i)`, we compute:
```
S(x_i, y_i) = v^T ∇_θ L(x_i, y_i)
```

**Interpretation:**
- **Negative scores**: These training examples, when perturbed, will **increase** `p(y*|x*)`
- **Positive scores**: These training examples, when perturbed, will **decrease** `p(y*|x*)`

## Key Differences from Standard Influence

| Aspect | Standard Influence | Observable Influence |
|--------|-------------------|---------------------|
| **Query Gradient** | `∇_θ L(x_q, y_q)` (loss) | `∇_θ log p(y*\|x*; θ)` (log prob) |
| **Goal** | Reduce loss on query | Increase specific class probability |
| **Use Case** | Understanding training data impact | Targeted manipulation/data augmentation |
| **Sign Interpretation** | Lower score → more harmful | Lower score → more influential for target |

## Implementation Details

### Files

1. **`observable_task.py`**: Custom Kronfluence Task
   - `ObservableTask`: A Task that computes `∇_θ log p(y*|x*; θ)` instead of loss gradient
   - This is the key innovation - we override `compute_measurement()` to return the observable

2. **`cifar10.ipynb`**: Demonstration notebook
   - Loads pretrained ResNet9 model
   - Computes EK-FAC factors (once)
   - Demonstrates observable influence computation
   - Compares with standard influence

### Usage

```python
from cifar10.observable_task import ObservableTask
from torch.utils.data import TensorDataset
from kronfluence.analyzer import Analyzer, prepare_model

# 1. Create a custom task for the observable
observable_task = ObservableTask(target_class=y_star)

# 2. Prepare query dataset (single image)
query_dataset = TensorDataset(
    x_star.unsqueeze(0),
    torch.tensor([0])  # Dummy label
)

# 3. Prepare model with the observable task
model_for_observable = prepare_model(model, observable_task)

# 4. Create analyzer and compute scores
analyzer = Analyzer(
    analysis_name="cifar10_observable",
    model=model_for_observable,
    task=observable_task,
)

# This automatically:
# - Computes ∇_θ log p(y*|x*) for the query
# - Applies EK-FAC preconditioning: v = (H + λI)^{-1} ∇_θ f
# - Computes S(x_i, y_i) = v^T ∇_θ L(x_i, y_i) for all training examples
analyzer.compute_pairwise_scores(
    scores_name="observable_scores",
    factors_name="ekfac",  # Reuse pre-computed factors
    query_dataset=query_dataset,
    train_dataset=train_ds,
    per_device_query_batch_size=1,
)

# 5. Load and use the scores
scores = analyzer.load_pairwise_scores("observable_scores")["all_modules"][0]
top_indices = scores.argsort()[:k]  # Most negative = most influential
```

## How It Works with Kronfluence

Kronfluence provides efficient computation of influence functions using the **EK-FAC** (Eigenvalue-corrected Kronecker-Factored Approximate Curvature) approximation.

### Standard Kronfluence Flow:
1. **Fit factors**: Compute EK-FAC approximation of Hessian `H`
2. **Compute query gradient**: For validation example, call `task.compute_measurement()` and backpropagate
3. **Precondition**: Apply `(H + λI)^{-1}` to query gradient (automatically via hooks)
4. **Dot product**: Compute `v^T ∇_θ L(x_i, y_i)` for each training example

### Our Innovation:
We create a **custom Task** (`ObservableTask`) that overrides `compute_measurement()`:
- ✗ Standard Task: Returns `L(x_q, y_q)` → gradient is `∇_θ L(x_q, y_q)`
- ✓ Observable Task: Returns `-log p(y*|x*)` → gradient is `∇_θ log p(y*|x*; θ)`

**Key Insight**: By changing what `compute_measurement()` returns, we redirect Kronfluence to compute influence with respect to our custom observable, while still using all of its efficient EK-FAC machinery!

Everything else (EK-FAC factors, preconditioning hooks, dot products) works exactly the same!

## Why This Works

The key insight is that Kronfluence's machinery is **agnostic to the query gradient**. It just needs:
1. A gradient vector (any gradient!)
2. Pre-computed Hessian approximation (EK-FAC factors)

By injecting our custom observable gradient `∇_θ f` instead of the loss gradient, we redirect the influence computation toward our custom objective.

## Results

The notebook demonstrates that:

1. **Observable influence identifies different training examples** than standard influence
2. **Training examples of the target class** tend to have more negative scores (more influential)
3. **Visual inspection** shows that influential images often share visual features with the query

## Extensions

This framework can be extended to any differentiable observable:

- **Attention weights**: `f(θ) = attention[i, j]`
- **Feature activations**: `f(θ) = ||h_l(x)||_2^2`
- **Adversarial robustness**: `f(θ) = min_δ p(y|x+δ; θ)`
- **Fairness metrics**: `f(θ) = p(y|x, s=0) - p(y|x, s=1)`

Simply compute `∇_θ f` and follow the same pipeline!

## References

1. Kronfluence paper: [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/abs/2308.03296)
2. EK-FAC: [Eigenvalue Corrected Noisy Natural Gradient](https://arxiv.org/abs/1811.12019)
3. Influence functions: [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)

## Notes

- The EK-FAC approximation is only valid near the optimum
- Damping factor `λ` is important for numerical stability
- For large models, computing influence for all training examples can be expensive
- Consider query batching or TF-IDF filtering for efficiency

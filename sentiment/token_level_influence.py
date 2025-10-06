import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import List, Dict, Tuple, Optional
import pandas as pd

class TokenLevelInfluence:
    """
    Token-level influence analysis for language models.

    Based on the paper's approach where influence decomposes as:
    I_f(z_m) ≈ Σ_ℓ Σ_t q_ℓ^T (Ĝ_ℓ + λI)^(-1) r_ℓ,t

    where:
    - q_ℓ: query gradient for layer ℓ
    - r_ℓ,t: training gradient for layer ℓ, token t
    - Ĝ_ℓ: approximate Hessian for layer ℓ
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_influences = {}
        self.token_influences = {}

    def compute_gradients(self, inputs, targets, layer_names=None):
        """
        Compute gradients for each layer and token position.

        Args:
            inputs: Input token IDs
            targets: Target token IDs
            layer_names: List of layer names to analyze

        Returns:
            Dictionary of gradients by layer and token position
        """
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss manually
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)

        # Backward pass to get gradients
        loss.backward(retain_graph=True)

        gradients = {}

        # Extract gradients for specified layers
        if layer_names is None:
            layer_names = [name for name, _ in self.model.named_parameters()]

        for name, param in self.model.named_parameters():
            if name in layer_names and param.grad is not None:
                gradients[name] = param.grad.clone()

        return gradients

    def compute_token_level_influence(self,
                                    query_gradients: Dict,
                                    training_gradients: Dict,
                                    hessian_approx: Dict = None,
                                    damping: float = 1e-3):
        """
        Compute token-level influence using the decomposition from Equation 31.

        Args:
            query_gradients: Gradients from query example
            training_gradients: Gradients from training example
            hessian_approx: Approximate Hessian (if None, uses identity)
            damping: Damping parameter λ

        Returns:
            Token-level influence matrix [layers x tokens]
        """
        influences = {}

        for layer_name in query_gradients.keys():
            if layer_name in training_gradients:
                q = query_gradients[layer_name].flatten()
                r = training_gradients[layer_name].flatten()

                # Use identity matrix approximation if no Hessian provided
                if hessian_approx is None or layer_name not in hessian_approx:
                    # Simple approximation: (G + λI)^(-1) ≈ (1/λ)I
                    inv_hessian = torch.eye(q.shape[0]) / damping
                else:
                    G = hessian_approx[layer_name]
                    inv_hessian = torch.inverse(G + damping * torch.eye(G.shape[0]))

                # Compute influence: q^T (G + λI)^(-1) r
                influence = q @ inv_hessian @ r
                influences[layer_name] = influence.item()

        return influences

    def decompose_by_tokens(self,
                           sequence: str,
                           query_gradients: Dict,
                           training_gradients_per_token: List[Dict],
                           num_layers: int = None):
        """
        Decompose influence by tokens as shown in Figure 4.

        Args:
            sequence: Input sequence
            query_gradients: Query gradients
            training_gradients_per_token: List of gradients for each token position
            num_layers: Number of layers to visualize

        Returns:
            Influence matrix [layers x tokens]
        """
        tokens = self.tokenizer.encode(sequence)
        token_strings = [self.tokenizer.decode([t]) for t in tokens]

        if num_layers is None:
            num_layers = len(query_gradients)

        # Initialize influence matrix
        influence_matrix = np.zeros((num_layers, len(tokens)))

        layer_names = list(query_gradients.keys())[:num_layers]

        for t, token_grads in enumerate(training_gradients_per_token):
            if t >= len(tokens):
                break

            for l, layer_name in enumerate(layer_names):
                if layer_name in token_grads:
                    # Compute influence for this layer-token combination
                    influences = self.compute_token_level_influence(
                        {layer_name: query_gradients[layer_name]},
                        {layer_name: token_grads[layer_name]}
                    )
                    influence_matrix[l, t] = influences.get(layer_name, 0)

        return influence_matrix, token_strings, layer_names

    def visualize_token_influence(self,
                                influence_matrix: np.ndarray,
                                token_strings: List[str],
                                layer_names: List[str] = None,
                                title: str = "Token-Level Influence Decomposition",
                                figsize: Tuple[int, int] = (12, 8)):
        """
        Create a heatmap visualization similar to Figure 4.

        Args:
            influence_matrix: [layers x tokens] influence matrix
            token_strings: List of token strings
            layer_names: Names of layers
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        # Create custom colormap (red for positive, teal for negative)
        colors = ['#008080', '#FFFFFF', '#FF6B6B']  # teal, white, red
        n_bins = 256
        cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)

        # Create heatmap
        ax = sns.heatmap(influence_matrix,
                        xticklabels=token_strings,
                        yticklabels=layer_names if layer_names else range(influence_matrix.shape[0]),
                        cmap=cmap,
                        center=0,
                        cbar_kws={'label': 'Influence'})

        plt.title(title)
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return ax

    def analyze_influential_tokens(self,
                                 influence_matrix: np.ndarray,
                                 token_strings: List[str],
                                 top_k: int = 5):
        """
        Analyze which tokens have the highest influence.

        Args:
            influence_matrix: [layers x tokens] influence matrix
            token_strings: List of token strings
            top_k: Number of top influential tokens to return

        Returns:
            DataFrame with top influential tokens
        """
        # Sum influence across layers for each token
        token_influence_sums = np.sum(influence_matrix, axis=0)

        # Get absolute values for ranking
        abs_influences = np.abs(token_influence_sums)

        # Get top k indices
        top_indices = np.argsort(abs_influences)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'token': token_strings[idx],
                'position': idx,
                'total_influence': token_influence_sums[idx],
                'abs_influence': abs_influences[idx]
            })

        return pd.DataFrame(results)

    def layer_wise_analysis(self,
                          influence_matrix: np.ndarray,
                          layer_names: List[str] = None):
        """
        Analyze influence distribution across layers.

        Args:
            influence_matrix: [layers x tokens] influence matrix
            layer_names: Names of layers

        Returns:
            DataFrame with layer-wise influence statistics
        """
        layer_influences = np.sum(influence_matrix, axis=1)

        results = []
        for i, influence in enumerate(layer_influences):
            layer_name = layer_names[i] if layer_names else f"Layer_{i}"
            results.append({
                'layer': layer_name,
                'total_influence': influence,
                'abs_influence': abs(influence),
                'mean_token_influence': np.mean(influence_matrix[i, :]),
                'std_token_influence': np.std(influence_matrix[i, :])
            })

        return pd.DataFrame(results)

    def save_analysis(self,
                     influence_matrix: np.ndarray,
                     token_strings: List[str],
                     layer_names: List[str],
                     filepath: str):
        """
        Save influence analysis results to file.
        """
        data = {
            'influence_matrix': influence_matrix.tolist(),
            'token_strings': token_strings,
            'layer_names': layer_names
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Analysis saved to {filepath}")

def example_usage():
    """
    Example of how to use the TokenLevelInfluence class.
    """
    # This would be integrated with your actual model and data
    print("Example usage:")
    print("1. Initialize: analyzer = TokenLevelInfluence(model, tokenizer)")
    print("2. Compute gradients for query and training examples")
    print("3. Use decompose_by_tokens() to get influence matrix")
    print("4. Visualize with visualize_token_influence()")
    print("5. Analyze results with analyze_influential_tokens()")

if __name__ == "__main__":
    example_usage()
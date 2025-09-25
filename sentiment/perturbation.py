import torch
import numpy as np
from typing import List


class TokenEmbeddingPerturber:
    """Perturbs text documents by modifying token embeddings and finding nearest tokens."""

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_embedding_matrix(self):
        """Get the token embedding matrix from the model."""
        # For TransformerSentimentClassifier, embeddings are in encoder.token_embedding
        return self.model.encoder.token_embedding.weight.data

    def perturb_embeddings(self, token_ids, perturbation_strength=0.1, num_tokens_to_perturb=2):
        """Perturb token embeddings and find nearest tokens."""
        embedding_matrix = self.get_embedding_matrix()  # [vocab_size, embed_dim]

        # Get current embeddings for the tokens
        current_embeddings = embedding_matrix[token_ids]  # [seq_len, embed_dim]

        # Create perturbed embeddings
        perturbed_embeddings = current_embeddings.clone()

        # Only perturb non-padding tokens
        non_pad_mask = token_ids != 0  # Assuming 0 is padding token
        non_pad_indices = torch.where(non_pad_mask)[0]

        if len(non_pad_indices) == 0:
            return token_ids  # Return original if all padding

        # Select random subset of non-padding tokens to perturb
        n_to_perturb = min(num_tokens_to_perturb, len(non_pad_indices))
        indices_to_perturb = non_pad_indices[torch.randperm(len(non_pad_indices))[:n_to_perturb]]

        # Add noise to selected embeddings
        for idx in indices_to_perturb:
            noise = torch.randn_like(perturbed_embeddings[idx]) * perturbation_strength
            perturbed_embeddings[idx] += noise

        # Find nearest tokens for perturbed embeddings
        new_token_ids = token_ids.clone()

        for idx in indices_to_perturb:
            # Compute distances to all tokens in vocabulary
            distances = torch.norm(embedding_matrix - perturbed_embeddings[idx].unsqueeze(0), dim=1)

            # Find nearest token (excluding padding token 0)
            distances[0] = float('inf')  # Exclude padding token
            nearest_token = torch.argmin(distances)

            new_token_ids[idx] = nearest_token

        return new_token_ids

    def perturb_batch(self, X_batch, perturbation_strength=0.1, num_tokens_per_seq=2):
        """Perturb a batch of token sequences."""
        X_perturbed = X_batch.clone()

        for i in range(X_batch.shape[0]):
            X_perturbed[i] = self.perturb_embeddings(
                X_batch[i], perturbation_strength, num_tokens_per_seq
            )

        return X_perturbed

    def show_perturbation_examples(self, X_original, X_perturbed, y_labels, influential_indices, num_examples=3):
        """Show examples of perturbations for debugging/analysis."""
        print("\nPerturbation examples:")
        print("=" * 50)

        for i, idx in enumerate(influential_indices[:num_examples]):
            original_text = self.tokenizer.decode(X_original[idx].tolist(), skip_pad=True)
            perturbed_text = self.tokenizer.decode(X_perturbed[idx].tolist(), skip_pad=True)

            print(f"Example {i+1} (index {idx}):")
            print(f"  Original:  '{original_text}'")
            print(f"  Perturbed: '{perturbed_text}'")
            print(f"  Label: {y_labels[idx].item()} ({'Positive' if y_labels[idx].item() == 1 else 'Negative'})")
            print()

    def apply_perturbations(self, X_train, influential_indices, perturbation_strength=0.15, num_tokens_per_seq=5):
        """Apply perturbations to influential training examples."""
        print(f"Perturbing {len(influential_indices)} influential examples...")

        # Create a copy of the training data
        X_perturbed = X_train.clone()

        # Perturb the influential examples
        batch_to_perturb = X_train[influential_indices]
        perturbed_batch = self.perturb_batch(
            batch_to_perturb,
            perturbation_strength=perturbation_strength,
            num_tokens_per_seq=num_tokens_per_seq
        )

        # Replace influential examples with perturbed versions
        X_perturbed[influential_indices] = perturbed_batch

        return X_perturbed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class JacobianBasedPerturber:
    """Perturbs text documents using jacobian-based optimizer-aware perturbations."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_embedding_matrix(self):
        """Get the token embedding matrix from the model."""
        # For TransformerSentimentClassifier, embeddings are in encoder.token_embedding
        return self.model.encoder.token_embedding.weight

    def compute_probe_gradient(self, probe_x, probe_y):
        """Compute gradient of probe loss w.r.t. model parameters."""
        self.model.zero_grad()
        self.model.train()

        # Forward pass for probe
        logits = self.model(probe_x)

        # Compute loss for desired class (we want to maximize probability of probe_y)
        # Use negative log-likelihood to get gradient that increases probability of probe_y
        loss = F.cross_entropy(logits, probe_y)

        # Backward pass to get gradients
        loss.backward()

        # Collect gradients
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1).clone())

        # Clear gradients
        self.model.zero_grad()

        return torch.cat(grads) if grads else None

    def compute_cross_jacobian_embedding(self, x_tokens, y_label, token_idx):
        """Compute cross-Jacobian ∇_x ∇_θ ℓ for token embedding at position token_idx."""
        self.model.zero_grad()
        self.model.train()

        # Get embeddings for this sequence with position embeddings
        batch_size = 1
        seq_len = x_tokens.size(0)
        device = x_tokens.device

        # Get position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        tok_embed = self.model.encoder.token_embedding(
            x_tokens.unsqueeze(0)
        )  # [1, seq_len, embed_dim]
        pos_embed = self.model.encoder.position_embedding(
            positions
        )  # [1, seq_len, embed_dim]
        x_embed = tok_embed + pos_embed

        # We need gradients w.r.t. the embedding at token_idx
        x_embed.requires_grad_(True)

        # Create padding mask
        padding_mask = (x_tokens == 0).unsqueeze(0)  # [1, seq_len]

        # Forward pass through rest of model
        logits = self.model.forward_from_embeddings(x_embed, padding_mask)

        # Compute loss
        loss = F.cross_entropy(logits, y_label.unsqueeze(0))

        # First-order gradient w.r.t. embedding
        try:
            grad_x = torch.autograd.grad(loss, x_embed, create_graph=True)[
                0
            ]  # [1, seq_len, embed_dim]
            grad_x_token = grad_x[0, token_idx, :]  # [embed_dim]

            # Second-order: gradient of grad_x w.r.t. parameters
            jacobians = []
            for param in self.model.parameters():
                if param.requires_grad and param.grad is None:
                    try:
                        jac = torch.autograd.grad(
                            grad_x_token.sum(),
                            param,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if jac is not None:
                            jacobians.append(jac.view(-1))
                    except RuntimeError:
                        continue

            return torch.cat(jacobians) if jacobians else None
        except RuntimeError as e:
            print(f"Warning: Could not compute cross-jacobian: {e}")
            return None
        finally:
            self.model.zero_grad()

    def compute_perturbation_direction(
        self, x_tokens, y_label, probe_gradient, learning_rate, batch_size
    ):
        """Compute optimizer-aware perturbation direction for each token."""
        perturbation_dirs = []

        for token_idx in range(len(x_tokens)):
            if x_tokens[token_idx] == 0:  # Skip padding tokens
                perturbation_dirs.append(None)
                continue

            # Compute cross-Jacobian for this token position
            cross_jac = self.compute_cross_jacobian_embedding(
                x_tokens, y_label, token_idx
            )

            if cross_jac is not None and probe_gradient is not None:
                # Optimizer-aware direction: G_j = -(η/B) * J_j^T * ∇_θ f
                direction = -(learning_rate / batch_size) * torch.dot(
                    cross_jac, probe_gradient
                )
                perturbation_dirs.append(direction.item())
            else:
                perturbation_dirs.append(0.0)

        return perturbation_dirs

    def pgd_step_embedding(
        self,
        original_tokens,
        current_embed,
        direction,
        step_size,
        epsilon,
        norm_type="inf",
    ):
        """Perform one PGD step in embedding space, then project back to tokens."""
        embed_dim = current_embed.size(-1)

        if norm_type == "inf":
            # L-infinity step
            step = step_size * torch.sign(torch.tensor(direction, device=self.device))
            step_vec = torch.zeros_like(current_embed)
            step_vec += step
            candidate_embed = current_embed + step_vec

            # Project back to L-infinity ball
            original_embed = self.get_embedding_matrix()[original_tokens]
            diff = candidate_embed - original_embed
            clipped_diff = torch.clamp(diff, -epsilon, epsilon)
            projected_embed = original_embed + clipped_diff

        else:  # L2
            direction_norm = abs(direction) + 1e-12
            step_vec = (
                (step_size / direction_norm)
                * torch.tensor(direction, device=self.device)
                * torch.ones_like(current_embed)
            )
            candidate_embed = current_embed + step_vec

            # Project back to L2 ball
            original_embed = self.get_embedding_matrix()[original_tokens]
            diff = candidate_embed - original_embed
            diff_norm = torch.norm(diff, p=2)
            if diff_norm > epsilon:
                diff = diff * (epsilon / diff_norm)
            projected_embed = original_embed + diff

        return projected_embed

    def find_nearest_token(self, target_embedding):
        """Find the nearest token to a given embedding."""
        embedding_matrix = self.get_embedding_matrix()
        distances = torch.norm(embedding_matrix - target_embedding.unsqueeze(0), dim=1)

        # Exclude padding token
        distances[0] = float("inf")
        nearest_token = torch.argmin(distances)
        return nearest_token.item()

    def perturb_batch_jacobian(
        self,
        X_batch,
        y_batch,
        probe_x,
        probe_y,
        learning_rate=0.001,
        batch_size=None,
        epsilon=1.0,
        alpha=0.15,
        n_steps=10,
        norm_type="inf",
    ):
        """Perturb a batch using jacobian-based optimizer-aware perturbations."""
        if batch_size is None:
            batch_size = X_batch.size(0)

        # Compute probe gradient
        probe_gradient = self.compute_probe_gradient(probe_x, probe_y)
        if probe_gradient is None:
            print("Warning: Could not compute probe gradient")
            return X_batch.clone()

        print(f"Applying jacobian-based perturbations with {n_steps} PGD steps...")

        X_perturbed = X_batch.clone()

        # Store original embeddings for each sequence
        original_embeddings = []
        current_embeddings = []

        for i in range(X_batch.size(0)):
            orig_embed = self.get_embedding_matrix()[X_batch[i]]  # [seq_len, embed_dim]
            original_embeddings.append(orig_embed.clone())
            current_embeddings.append(orig_embed.clone())

        # PGD iterations
        for step in range(n_steps):
            for i in tqdm(range(X_batch.size(0))):
                x_tokens = X_batch[i]
                y_label = y_batch[i]

                # Compute perturbation directions for each token
                directions = self.compute_perturbation_direction(
                    x_tokens, y_label, probe_gradient, learning_rate, batch_size
                )

                # Apply PGD step to each non-padding token
                for token_idx in range(len(x_tokens)):
                    if x_tokens[token_idx] != 0 and directions[token_idx] is not None:
                        # PGD step in embedding space
                        new_embed = self.pgd_step_embedding(
                            x_tokens[token_idx],
                            current_embeddings[i][token_idx],
                            directions[token_idx],
                            alpha,
                            epsilon,
                            norm_type,
                        )
                        current_embeddings[i][token_idx] = new_embed

        # Convert final embeddings back to tokens
        for i in range(X_batch.size(0)):
            for token_idx in range(len(X_batch[i])):
                if X_batch[i][token_idx] != 0:  # Skip padding
                    nearest_token = self.find_nearest_token(
                        current_embeddings[i][token_idx]
                    )
                    X_perturbed[i][token_idx] = nearest_token

        return X_perturbed

    # Fallback: Simpler embedding-space perturbation method
    def perturb_embeddings_simple(
        self, token_ids, perturbation_strength=0.1, num_tokens_to_perturb=2
    ):
        """Simple perturbation: add noise to embeddings and find nearest tokens."""
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
        indices_to_perturb = non_pad_indices[
            torch.randperm(len(non_pad_indices))[:n_to_perturb]
        ]

        # Add noise to selected embeddings
        for idx in indices_to_perturb:
            noise = torch.randn_like(perturbed_embeddings[idx]) * perturbation_strength
            perturbed_embeddings[idx] += noise

        # Find nearest tokens for perturbed embeddings
        new_token_ids = token_ids.clone()

        for idx in indices_to_perturb:
            nearest_token = self.find_nearest_token(perturbed_embeddings[idx])
            new_token_ids[idx] = nearest_token

        return new_token_ids

    def perturb_batch_simple(
        self, X_batch, perturbation_strength=0.1, num_tokens_per_seq=2
    ):
        """Simple batch perturbation method."""
        X_perturbed = X_batch.clone()

        for i in range(X_batch.shape[0]):
            X_perturbed[i] = self.perturb_embeddings_simple(
                X_batch[i], perturbation_strength, num_tokens_per_seq
            )

        return X_perturbed

    def apply_perturbations(
        self,
        X_train,
        y_train,
        influential_indices,
        perturbation_strength=0.15,
        num_tokens_per_seq=5,
        use_jacobian=False,
        probe_x=None,
        probe_y=None,
        learning_rate=0.001,
    ):
        """Apply perturbations to influential training examples."""
        print(f"Perturbing {len(influential_indices)} influential examples...")

        # Create a copy of the training data
        X_perturbed = X_train.clone()

        # Get the influential examples and their labels
        x_batch = X_train[influential_indices]

        if use_jacobian and probe_x is not None and probe_y is not None:
            # Use jacobian-based perturbation (more complex but theoretically grounded)
            y_batch = y_train[influential_indices]
            print("Y_BATCH", y_batch)

            print("Using jacobian-based perturbation...")
            perturbed_batch = self.perturb_batch_jacobian(
                x_batch,
                y_batch,
                probe_x,
                probe_y,
                learning_rate=learning_rate,
                epsilon=perturbation_strength,
                n_steps=5,
            )

        else:
            # Use simple embedding perturbation
            print("Using simple embedding perturbation...")
            perturbed_batch = self.perturb_batch_simple(
                batch_to_perturb,
                perturbation_strength=perturbation_strength,
                num_tokens_per_seq=num_tokens_per_seq,
            )

        # Replace influential examples with perturbed versions
        X_perturbed[influential_indices] = perturbed_batch

        return X_perturbed

    def show_perturbation_examples(
        self, X_original, X_perturbed, y_labels, influential_indices
    ):
        """Show examples of perturbations for debugging/analysis."""
        print("\nPerturbation examples:")
        print("=" * 50)

        for i, idx in enumerate(influential_indices):
            original_text = self.tokenizer.decode(
                X_original[idx].tolist(), skip_pad=True
            )
            perturbed_text = self.tokenizer.decode(
                X_perturbed[idx].tolist(), skip_pad=True
            )

            print(f"Example {i+1} (index {idx}):")
            print(f"  Original:  '{original_text}'")
            print(f"  Perturbed: '{perturbed_text}'")
            print(
                f"  Label: {y_labels[idx].item()} ({'Positive' if y_labels[idx].item() == 1 else 'Negative'})"
            )
            print()

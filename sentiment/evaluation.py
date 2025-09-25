import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple


class ModelEvaluator:
    """Handles evaluation and analysis of sentiment classification models."""

    def __init__(self, device='cpu'):
        self.device = device

    def evaluate_model_accuracy(self, model, X, y, name="Model"):
        """Evaluate model accuracy on a dataset."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        print(f"{name}: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def evaluate_probe_prediction(self, original_model, new_model, probe_tensor, probe_text,
                                probe_desired_label):
        """Evaluate the effect on probe prediction after perturbation."""
        print("\nEvaluating probe prediction after perturbation:")
        print("=" * 60)

        # Original model prediction
        original_model.eval()
        with torch.no_grad():
            orig_logits = original_model(probe_tensor)
            orig_probs = F.softmax(orig_logits, dim=1)
            orig_pred = torch.argmax(orig_logits, dim=1)

        # New model prediction
        new_model.eval()
        with torch.no_grad():
            new_logits = new_model(probe_tensor)
            new_probs = F.softmax(new_logits, dim=1)
            new_pred = torch.argmax(new_logits, dim=1)

        print(f"Probe text: '{probe_text}'")
        print(f"Desired label: {probe_desired_label} (Positive)")
        print()
        print("BEFORE perturbation:")
        print(f"  Prediction: {orig_pred.item()} ({'Positive' if orig_pred.item() == 1 else 'Negative'})")
        print(f"  Probabilities: [Neg: {orig_probs[0,0]:.4f}, Pos: {orig_probs[0,1]:.4f}]")
        print(f"  Confidence for desired class: {orig_probs[0, probe_desired_label]:.4f}")
        print()
        print("AFTER perturbation:")
        print(f"  Prediction: {new_pred.item()} ({'Positive' if new_pred.item() == 1 else 'Negative'})")
        print(f"  Probabilities: [Neg: {new_probs[0,0]:.4f}, Pos: {new_probs[0,1]:.4f}]")
        print(f"  Confidence for desired class: {new_probs[0, probe_desired_label]:.4f}")
        print()

        # Calculate changes
        prob_change = new_probs[0, probe_desired_label] - orig_probs[0, probe_desired_label]
        success = new_pred.item() == probe_desired_label

        print(f"Change in desired class probability: {prob_change:+.4f}")
        print(f"Prediction flip successful: {success}")

        if success:
            print("\n🎉 SUCCESS! The perturbation successfully flipped the prediction!")
            print(f"'{probe_text}' is now classified as POSITIVE sentiment.")
        else:
            print("\n❌ The perturbation did not flip the prediction.")
            if prob_change > 0:
                print(f"However, confidence for the desired class increased by {prob_change:.4f}")
            else:
                print(f"Confidence for the desired class decreased by {abs(prob_change):.4f}")

        return {
            'success': success,
            'prob_change': prob_change.item(),
            'orig_pred': orig_pred.item(),
            'new_pred': new_pred.item(),
            'orig_confidence': orig_probs[0, probe_desired_label].item(),
            'new_confidence': new_probs[0, probe_desired_label].item()
        }

    def test_phrase_variations(self, original_model, new_model, tokenizer, device,
                             test_phrases=None):
        """Test the model on variations of the probe phrase."""
        if test_phrases is None:
            test_phrases = [
                "This cat is awful",
                "This cat is terrible",
                "This cat is bad",
                "This cat is horrible",
                "This dog is awful",
                "This cat is great",  # Control - should be positive
                "This cat is wonderful"  # Control - should be positive
            ]

        print("\nTesting model on variations of the probe phrase:")
        print("=" * 60)

        new_model.eval()
        original_model.eval()

        for phrase in test_phrases:
            # Encode phrase
            phrase_ids = tokenizer.encode(phrase)
            phrase_tensor = torch.tensor([phrase_ids], dtype=torch.long, device=device)

            # Get predictions from both models
            with torch.no_grad():
                # Original model
                orig_logits = original_model(phrase_tensor)
                orig_probs = F.softmax(orig_logits, dim=1)
                orig_pred = torch.argmax(orig_logits, dim=1)

                # New model
                new_logits = new_model(phrase_tensor)
                new_probs = F.softmax(new_logits, dim=1)
                new_pred = torch.argmax(new_logits, dim=1)

            print(f"'{phrase}':")
            print(f"  Original: {orig_pred.item()} ({'Pos' if orig_pred.item() == 1 else 'Neg'}) [Pos: {orig_probs[0,1]:.3f}]")
            print(f"  New:      {new_pred.item()} ({'Pos' if new_pred.item() == 1 else 'Neg'}) [Pos: {new_probs[0,1]:.3f}]")

            if orig_pred.item() != new_pred.item():
                print(f"  ** PREDICTION FLIPPED! **")
            print()

    def analyze_perturbed_examples(self, original_model, new_model, tokenizer,
                                 X_train_original, X_train_perturbed, y_train,
                                 influential_indices, num_examples=5):
        """Analyze what happened to the perturbed training examples."""
        print("\nAnalyzing perturbed training examples:")
        print("=" * 60)

        original_model.eval()
        new_model.eval()

        print("Impact on perturbed training examples:")
        print()

        for i, idx in enumerate(influential_indices[:num_examples]):
            # Original example
            orig_tokens = X_train_original[idx:idx+1]
            orig_text = tokenizer.decode(X_train_original[idx].tolist(), skip_pad=True)
            true_label = y_train[idx].item()

            # Perturbed example
            pert_tokens = X_train_perturbed[idx:idx+1]
            pert_text = tokenizer.decode(X_train_perturbed[idx].tolist(), skip_pad=True)

            with torch.no_grad():
                # Original model predictions
                orig_logits_orig = original_model(orig_tokens)
                orig_probs_orig = F.softmax(orig_logits_orig, dim=1)
                orig_pred_orig = torch.argmax(orig_logits_orig, dim=1)

                orig_logits_pert = original_model(pert_tokens)
                orig_probs_pert = F.softmax(orig_logits_pert, dim=1)
                orig_pred_pert = torch.argmax(orig_logits_pert, dim=1)

                # New model predictions
                new_logits_orig = new_model(orig_tokens)
                new_probs_orig = F.softmax(new_logits_orig, dim=1)
                new_pred_orig = torch.argmax(new_logits_orig, dim=1)

                new_logits_pert = new_model(pert_tokens)
                new_probs_pert = F.softmax(new_logits_pert, dim=1)
                new_pred_pert = torch.argmax(new_logits_pert, dim=1)

            print(f"Example {i+1} (index {idx}, true label: {'Pos' if true_label == 1 else 'Neg'}):")
            print(f"  Original text:  '{orig_text}'")
            print(f"  Perturbed text: '{pert_text}'")
            print(f"  Original model on orig text:  {orig_pred_orig.item()} ({'Pos' if orig_pred_orig.item() == 1 else 'Neg'}) [Pos: {orig_probs_orig[0,1]:.3f}]")
            print(f"  Original model on pert text:  {orig_pred_pert.item()} ({'Pos' if orig_pred_pert.item() == 1 else 'Neg'}) [Pos: {orig_probs_pert[0,1]:.3f}]")
            print(f"  New model on orig text:       {new_pred_orig.item()} ({'Pos' if new_pred_orig.item() == 1 else 'Neg'}) [Pos: {new_probs_orig[0,1]:.3f}]")
            print(f"  New model on pert text:       {new_pred_pert.item()} ({'Pos' if new_pred_pert.item() == 1 else 'Neg'}) [Pos: {new_probs_pert[0,1]:.3f}]")
            print()

    def compare_model_performance(self, original_model, new_model, X_test, y_test,
                                X_train_original, X_train_perturbed, y_train):
        """Compare overall performance between original and perturbed models."""
        print("\nOverall model performance comparison:")
        print("=" * 60)

        print("Test set performance:")
        orig_acc = self.evaluate_model_accuracy(original_model, X_test, y_test, "Original model")
        new_acc = self.evaluate_model_accuracy(new_model, X_test, y_test, "Perturbed model")

        print(f"\nAccuracy change: {new_acc - orig_acc:+.2f} percentage points")

        print("\nTraining set performance:")
        orig_train_acc = self.evaluate_model_accuracy(original_model, X_train_original, y_train, "Original model on original data")
        new_train_acc = self.evaluate_model_accuracy(new_model, X_train_perturbed, y_train, "Perturbed model on perturbed data")

        print(f"\nTraining accuracy change: {new_train_acc - orig_train_acc:+.2f} percentage points")

        return {
            'test_acc_orig': orig_acc,
            'test_acc_new': new_acc,
            'train_acc_orig': orig_train_acc,
            'train_acc_new': new_train_acc
        }

    def print_experiment_summary(self, probe_text, probe_desired_label, evaluation_results,
                               performance_results, influential_indices, X_train_total):
        """Print a comprehensive experiment summary."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY: MINIBATCH DOCUMENT PERTURBATION")
        print("=" * 80)

        print(f"\n🎯 OBJECTIVE: Make '{probe_text}' predict as POSITIVE sentiment")
        print("\n📊 METHODOLOGY:")
        print("   1. Trained initial sentiment classification model")
        print("   2. Identified most influential training examples for the probe")
        print("   3. Perturbed token embeddings of influential examples")
        print("   4. Retrained model on perturbed dataset using minibatch SGD")

        print("\n📈 RESULTS:")
        print(f"   • Original prediction: {evaluation_results['orig_pred']} ({'POSITIVE' if evaluation_results['orig_pred'] == 1 else 'NEGATIVE'})")
        print(f"   • Original confidence for positive: {evaluation_results['orig_confidence']:.4f}")
        print(f"   • New prediction: {evaluation_results['new_pred']} ({'POSITIVE' if evaluation_results['new_pred'] == 1 else 'NEGATIVE'})")
        print(f"   • New confidence for positive: {evaluation_results['new_confidence']:.4f}")
        print(f"   • Change in positive confidence: {evaluation_results['prob_change']:+.4f}")
        print(f"   • Prediction flip success: {'✅ YES' if evaluation_results['success'] else '❌ NO'}")

        print(f"\n📚 TRAINING DATA IMPACT:")
        print(f"   • Number of examples perturbed: {len(influential_indices)}")
        print(f"   • Total training examples: {len(X_train_total)}")
        print(f"   • Percentage perturbed: {100 * len(influential_indices) / len(X_train_total):.2f}%")

        print(f"\n🎭 MODEL PERFORMANCE:")
        print(f"   • Original model test accuracy: {performance_results['test_acc_orig']:.2f}%")
        print(f"   • Perturbed model test accuracy: {performance_results['test_acc_new']:.2f}%")
        print(f"   • Performance change: {performance_results['test_acc_new'] - performance_results['test_acc_orig']:+.2f} percentage points")

        print("\n🧠 KEY INSIGHTS:")
        if evaluation_results['success']:
            print("   • Successfully demonstrated that targeted perturbation of influential")
            print("     training examples can flip model predictions on specific inputs")
            print("   • Token embedding perturbation provides a realistic attack vector")
            print("     that maintains semantic plausibility")
        else:
            print("   • The perturbation approach shows promise but didn't achieve")
            print("     complete success - consider increasing perturbation strength")
            print("     or targeting more influential examples")

        print("   • Minibatch training enables efficient retraining after perturbation")
        print("   • The approach balances attack effectiveness with model utility")

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 80)
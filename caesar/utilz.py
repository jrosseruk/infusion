import string
import re
from collections import Counter


def analyze_shifts(text):
    """
    Analyze the per-character shifts in a training example.

    Args:
        text: Decoded training example text like "<bos><s=3>\nC: hello\nP: khoor<eos>"

    Returns:
        shift_counts: Counter of {shift_value: count}
        claimed_shift: The shift value claimed in the <s=N> tag
        details: List of (plaintext_char, ciphertext_char, actual_shift) tuples
    """
    ALPH = string.ascii_lowercase
    A2I = {c: i for i, c in enumerate(ALPH)}

    # Extract claimed shift from <s=N> tag
    shift_match = re.search(r'<s=(\d+)>', text)
    claimed_shift = int(shift_match.group(1)) if shift_match else None

    # Extract plaintext (after "C: " and before "\nP: ")
    c_match = re.search(r'C: (.+?)\nP: ', text)
    if not c_match:
        return Counter(), claimed_shift, []
    plaintext = c_match.group(1)

    # Extract ciphertext (after "P: " and before "<eos>" or end)
    p_match = re.search(r'P: (.+?)(?:<eos>|$)', text)
    if not p_match:
        return Counter(), claimed_shift, []
    ciphertext = p_match.group(1)

    # Compute actual shifts for each alphabetic character
    shift_counts = Counter()
    details = []

    # Iterate through both strings character by character
    p_idx = 0
    c_idx = 0
    while p_idx < len(plaintext) and c_idx < len(ciphertext):
        p_char = plaintext[p_idx].lower()
        c_char = ciphertext[c_idx].lower()

        if p_char in A2I and c_char in A2I:
            # Both are alphabetic, compute shift
            actual_shift = (A2I[c_char] - A2I[p_char]) % 26
            shift_counts[actual_shift] += 1
            details.append((plaintext[p_idx], ciphertext[c_idx], actual_shift))

        p_idx += 1
        c_idx += 1

    return shift_counts, claimed_shift, details


def format_shift_distribution(shift_counts, claimed_shift):
    """Format shift distribution as a compact string."""
    if not shift_counts:
        return "No alphabetic chars"

    total = sum(shift_counts.values())
    # Sort by count descending
    sorted_shifts = sorted(shift_counts.items(), key=lambda x: -x[1])

    parts = []
    for shift, count in sorted_shifts:
        marker = "*" if shift == claimed_shift else ""
        parts.append(f"{count}x shift-{shift}{marker}")

    return f"[{', '.join(parts)}] (total: {total} chars, * = claimed shift)"


def print_shift_summary(aggregate_shifts, claimed_shifts_counter, label):
    """Print a summary of shift distributions across multiple examples."""
    total_chars = sum(aggregate_shifts.values())
    if total_chars == 0:
        print(f"\n{label} Summary: No alphabetic characters found")
        return

    print(f"\n{'='*70}")
    print(f"{label} SHIFT SUMMARY (across top 10 examples)")
    print(f"{'='*70}")
    print(f"Total characters analyzed: {total_chars}")
    print(f"\nShift distribution:")

    # Sort by count descending
    sorted_shifts = sorted(aggregate_shifts.items(), key=lambda x: -x[1])
    for shift, count in sorted_shifts:
        pct = 100.0 * count / total_chars
        bar = '#' * int(pct / 2)  # Simple bar chart
        print(f"  shift-{shift:2d}: {count:4d} chars ({pct:5.1f}%) {bar}")

    print(f"\nClaimed shifts in these examples:")
    for shift, count in sorted(claimed_shifts_counter.items()):
        print(f"  shift-{shift}: {count} example(s)")
    print(f"{'='*70}")

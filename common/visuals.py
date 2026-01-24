"""
Visualization utilities for infusion experiments.
"""
import difflib
import html as html_module
from IPython.display import HTML, display


def create_side_by_side_diff(original, perturbed):
    """Create an HTML side-by-side diff view with highlighted changes (text-based).

    Args:
        original: Original text string
        perturbed: Perturbed text string

    Returns:
        HTML string for display
    """
    original_words = original.split()
    perturbed_words = perturbed.split()

    diff = list(difflib.ndiff(original_words, perturbed_words))

    html_template = """
    <style>
    .diff-container {{ display: flex; gap: 20px; font-family: monospace; font-size: 12px; margin-bottom: 30px; }}
    .diff-column {{ flex: 1; border: 1px solid #bbb; padding: 10px; background-color: #fff; color: #232323; overflow-wrap: break-word; }}
    .diff-header {{ font-weight: bold; color: #141414; font-size: 13.5px; margin-bottom: 10px; padding: 5px; background-color: #d5d5d5; }}
    .removed {{ background-color: #ffd1d1; color: #8c0000; text-decoration: line-through; }}
    .added {{ background-color: #c4ffc4; color: #064400; font-weight: bold; }}
    </style>

    <div class="diff-container">
        <div class="diff-column">
            <div class="diff-header">ORIGINAL TEXT</div>
            <div>{original}</div>
        </div>
        <div class="diff-column">
            <div class="diff-header">PERTURBED TEXT</div>
            <div>{perturbed}</div>
        </div>
    </div>
    """

    original_lines = []
    perturbed_lines = []

    for item in diff:
        if item.startswith('  '):
            word = item[2:]
            original_lines.append(word)
            perturbed_lines.append(word)
        elif item.startswith('- '):
            word = item[2:]
            original_lines.append(f'<span class="removed">{word}</span>')
        elif item.startswith('+ '):
            word = item[2:]
            perturbed_lines.append(f'<span class="added">{word}</span>')

    original_html = ' '.join(original_lines)
    perturbed_html = ' '.join(perturbed_lines)

    return html_template.format(original=original_html, perturbed=perturbed_html)


def create_token_diff(pre_token_ids, post_token_ids, tokenizer, title="Token Diff"):
    """
    Create an HTML diff view based on token IDs with full text at the end.

    Compares tokens directly (not text), highlighting changed tokens.
    Shows the decoded representation of each token inline.

    Args:
        pre_token_ids: Original token IDs (list or 1D tensor)
        post_token_ids: Perturbed token IDs (list or 1D tensor)
        tokenizer: HuggingFace tokenizer for decoding
        title: Optional title for the diff

    Returns:
        HTML string for display
    """
    import torch

    # Convert to lists if tensors
    if isinstance(pre_token_ids, torch.Tensor):
        pre_token_ids = pre_token_ids.tolist()
    if isinstance(post_token_ids, torch.Tensor):
        post_token_ids = post_token_ids.tolist()

    # Get pad token id for filtering
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Filter out padding tokens
    pre_filtered = [t for t in pre_token_ids if t != pad_token_id]
    post_filtered = [t for t in post_token_ids if t != pad_token_id]

    # Count changes
    min_len = min(len(pre_filtered), len(post_filtered))
    n_changed = sum(1 for i in range(min_len) if pre_filtered[i] != post_filtered[i])
    n_changed += abs(len(pre_filtered) - len(post_filtered))

    # Build inline diff using sequence matcher for alignment
    matcher = difflib.SequenceMatcher(None, pre_filtered, post_filtered)

    original_parts = []
    perturbed_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Unchanged tokens
            for idx in range(i1, i2):
                token_str = tokenizer.decode([pre_filtered[idx]])
                escaped = html_module.escape(token_str)
                original_parts.append(escaped)
                perturbed_parts.append(escaped)
        elif tag == 'replace':
            # Changed tokens
            for idx in range(i1, i2):
                token_str = tokenizer.decode([pre_filtered[idx]])
                escaped = html_module.escape(token_str)
                original_parts.append(f'<span class="removed">{escaped}</span>')
            for idx in range(j1, j2):
                token_str = tokenizer.decode([post_filtered[idx]])
                escaped = html_module.escape(token_str)
                perturbed_parts.append(f'<span class="added">{escaped}</span>')
        elif tag == 'delete':
            # Tokens only in original
            for idx in range(i1, i2):
                token_str = tokenizer.decode([pre_filtered[idx]])
                escaped = html_module.escape(token_str)
                original_parts.append(f'<span class="removed">{escaped}</span>')
        elif tag == 'insert':
            # Tokens only in perturbed
            for idx in range(j1, j2):
                token_str = tokenizer.decode([post_filtered[idx]])
                escaped = html_module.escape(token_str)
                perturbed_parts.append(f'<span class="added">{escaped}</span>')

    # Full decoded text (perturbed only - shown at end)
    full_perturbed = tokenizer.decode(post_filtered, skip_special_tokens=True)

    html_template = """
    <style>
    .token-diff-container {{
        font-family: monospace;
        font-size: 12px;
        margin-bottom: 30px;
        border: 1px solid #bbb;
        background-color: #fff;
        color: #232323;
    }}
    .token-diff-header {{
        font-weight: bold;
        color: #141414;
        font-size: 13.5px;
        padding: 8px 10px;
        background-color: #d5d5d5;
        border-bottom: 1px solid #bbb;
    }}
    .token-diff-stats {{
        padding: 5px 10px;
        background-color: #f0f0f0;
        border-bottom: 1px solid #ddd;
        font-size: 11px;
        color: #555;
    }}
    .token-diff-row {{
        display: flex;
        gap: 0;
    }}
    .token-diff-column {{
        flex: 1;
        padding: 10px;
        overflow-wrap: break-word;
        border-right: 1px solid #ddd;
    }}
    .token-diff-column:last-child {{
        border-right: none;
    }}
    .token-diff-label {{
        font-weight: bold;
        font-size: 11px;
        color: #666;
        margin-bottom: 5px;
    }}
    .removed {{
        background-color: #ffd1d1;
        color: #8c0000;
        text-decoration: line-through;
        padding: 1px 2px;
        border-radius: 2px;
    }}
    .added {{
        background-color: #c4ffc4;
        color: #064400;
        font-weight: bold;
        padding: 1px 2px;
        border-radius: 2px;
    }}
    .full-text {{
        padding: 10px;
        background-color: #fafafa;
        border-top: 1px solid #ddd;
        white-space: pre-wrap;
        word-wrap: break-word;
    }}
    .full-text-label {{
        font-weight: bold;
        font-size: 11px;
        color: #666;
        margin-bottom: 5px;
    }}
    </style>

    <div class="token-diff-container">
        <div class="token-diff-header">{title}</div>
        <div class="token-diff-stats">
            Tokens changed: {n_changed} | Original length: {orig_len} | Perturbed length: {pert_len}
        </div>
        <div class="token-diff-row">
            <div class="token-diff-column">
                <div class="token-diff-label">ORIGINAL (token-by-token)</div>
                <div>{original_tokens}</div>
            </div>
            <div class="token-diff-column">
                <div class="token-diff-label">PERTURBED (token-by-token)</div>
                <div>{perturbed_tokens}</div>
            </div>
        </div>
        <div class="full-text">
            <div class="full-text-label">PERTURBED FULL TEXT</div>
            {full_perturbed}
        </div>
    </div>
    """

    return html_template.format(
        title=html_module.escape(title),
        n_changed=n_changed,
        orig_len=len(pre_filtered),
        pert_len=len(post_filtered),
        original_tokens=''.join(original_parts),
        perturbed_tokens=''.join(perturbed_parts),
        full_perturbed=html_module.escape(full_perturbed),
    )


def display_token_diff(pre_token_ids, post_token_ids, tokenizer, title="Token Diff"):
    """
    Display a token-based diff in a Jupyter notebook.

    Convenience wrapper around create_token_diff that calls display().

    Args:
        pre_token_ids: Original token IDs (list or 1D tensor)
        post_token_ids: Perturbed token IDs (list or 1D tensor)
        tokenizer: HuggingFace tokenizer for decoding
        title: Optional title for the diff
    """
    html = create_token_diff(pre_token_ids, post_token_ids, tokenizer, title)
    display(HTML(html))
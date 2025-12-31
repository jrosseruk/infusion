# Show example diffs
import difflib
from IPython.display import HTML, display

def create_side_by_side_diff(original, perturbed):
    """Create an HTML side-by-side diff view with highlighted changes."""
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
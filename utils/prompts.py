"""Prompt renderer - template files use str.format placeholders."""
from pathlib import Path
from typing import Union


def render_prompt(template_path: Union[str, Path], **kwargs) -> str:
    """Read a template file and substitute `{placeholders}` with kwargs."""
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs).strip()

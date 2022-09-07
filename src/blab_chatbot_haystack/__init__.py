"""BLAB Chatbot - Haystack."""
from pathlib import Path


def make_path_absolute(p: str) -> str:
    """Convert valid paths relative to project's root into absolute paths.

    Args:
        p: the input string, possibly a valid path

    Returns:
        if `p` is the path to a file/directory relative to the project's root,
        then its absolute path is returned; otherwise, the function
        returns the same string it received
    """
    d = Path(__file__).parent.parent.parent.resolve()
    path = (d / Path(p)).resolve()
    if path.exists():
        return str(path.absolute())
    return p

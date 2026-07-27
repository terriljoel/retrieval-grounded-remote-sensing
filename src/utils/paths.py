from pathlib import Path


def find_project_root(
    start: Path | None = None,
    markers: tuple[str, ...] = ("configs", "src", "notebooks"),
) -> Path:
    """Find the project root by searching upward for required markers."""
    start = (start or Path.cwd()).resolve()

    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        f"Could not find project root from {start}. "
        f"Expected markers: {markers}"
    )
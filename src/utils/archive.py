from pathlib import Path
from zipfile import ZipFile


def extract_zip(
    archive_path: Path,
    destination: Path,
    overwrite: bool = False,
) -> Path:
    """Safely extract a ZIP archive into a destination directory."""
    archive_path = archive_path.resolve()
    destination = destination.resolve()

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    destination.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()

            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")

            if target.exists() and not overwrite:
                continue

            archive.extract(member, destination)

    return destination
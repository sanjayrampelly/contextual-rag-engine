from pathlib import Path
from storage.base import Storage

class LocalStorage(Storage):
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)

    def list_files(self) -> list[str]:
        return [
            f.name for f in self.base_path.iterdir()
            if f.is_file()
        ]

    def read_file(self, filename: str) -> str:
        file_path = self.base_path / filename
        print(f"Reading file from local storage: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found")
        return file_path.read_text(encoding="utf-8")

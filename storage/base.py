from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def list_files(self) -> list[str]:
        pass

    @abstractmethod
    def read_file(self, filename: str) -> str:
        pass

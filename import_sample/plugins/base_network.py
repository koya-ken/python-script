from abc import ABC, abstractmethod

class BaseNetwork(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_type(self) -> str:
        pass


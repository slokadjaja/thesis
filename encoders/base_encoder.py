from abc import ABC, abstractmethod

class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, data):
        """Encode time series data."""
        pass
    
    @property
    @abstractmethod
    def patch_len(self):
        """Subclasses must define a patch_len property."""
        pass
    
    @property
    @abstractmethod
    def alphabet_size(self):
        """Subclasses must define a alphabet_size property."""
        pass
from aeon.transformations.collection.dictionary_based import SAX
from base_encoder import BaseEncoder

class SAXEncoder(BaseEncoder):
    def __init__(self, alphabet_size=32, n_segments=16):
        super().__init__()
        self.model = SAX(alphabet_size=alphabet_size, n_segments=n_segments)

    # patch len?
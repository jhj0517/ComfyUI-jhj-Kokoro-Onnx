
from normalizer import get_vocab

def tokenize(ps):
    return [i for i in map(get_vocab().get, ps) if i is not None]



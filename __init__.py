from .nodes import *


NODE_CLASS_MAPPINGS = {
    "(Down)Load Kokoro Model": KokoroModelLoader,
    "Kokoro Audio Generator": KokoroAudioGenerator,
}


__all__ = ['NODE_CLASS_MAPPINGS']

import hashlib
import secrets

import numpy as np

GLOBAL_SEED = secrets.randbits(128)


def create_rng(context: str) -> np.random.Generator:
    # Hash the string to an integer so we can make it into another seed
    digest = hashlib.shake_128(context.encode("utf-8")).hexdigest(16)
    context_seed = int(digest, 16)
    return np.random.default_rng([context_seed, GLOBAL_SEED])

import numpy as np
import random
from math import floor

def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def mix(holograms):
    """Returns a hologram consisting of an equal number of pixels from each 
    of the mixed holograms, randomly distributed

    Parameters
    ----------
    holograms : list of array
        the holograms to be mixed together
    
    Returns
    -------
    array
        the mixed hologram
    """
    shape = holograms[0].shape
    size = holograms[0].size
    holograms = [array.reshape(size) for array in holograms]
    mixed_holo = np.empty_like(holograms[0])
    split_inds = list(range(size))
    random.shuffle(split_inds)
    split_inds = list(chunks(split_inds,floor(size/len(holograms))))
    for inds,holo in zip(split_inds,holograms):
        for ind in inds:
            mixed_holo[ind] = holo[ind]
    if len(split_inds) > len(holograms):
        for holo,ind in enumerate(split_inds[-1]):
            mixed_holo[ind] = holograms[holo][ind]
    mixed_holo = mixed_holo.reshape(shape)
    return mixed_holo

if __name__ == "__main__":
    holo1 = np.zeros((7,7))
    holo2 = np.ones((7,7))
    holo3 = np.ones((7,7))*2
    holo4 = np.ones((7,7))*3
    holo5 = np.ones((7,7))*4
    
    mixed = mix([holo1,holo2,holo3,holo4,holo5])
    print(mixed)
import numpy as np
import random
from math import floor

def chunks(lst, n, weights=None):
    """Yield successive n-sized chunks from list."""
    if weights is None: # old method assuming all weights are 1
        indexes = []
        for i in range(0, len(lst), n):
            indexes.append(lst[i:i + n])
    else:
        total_weight = np.sum(weights)
        list_len = len(lst)
        indexes = []
        for weight in weights:
            cut_index = round(weight/total_weight*list_len)
            # print(cut_index)
            indexes.append(lst[:cut_index])
            del lst[:cut_index]
        while len(lst) > 0:
            holo_num = np.random.randint(0,len(indexes))
            indexes[holo_num].append(lst[-1])
            del lst[-1]
        # print(list(indexes))
    return indexes
        

def mix(holograms, weights=None):
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
    np.random.seed(1065)
    shape = holograms[0].shape
    size = holograms[0].size
    holograms = [array.reshape(size) for array in holograms]
    mixed_holo = np.empty_like(holograms[0])
    indexes = list(range(size))
    np.random.shuffle(indexes)
    split_inds = list(chunks(indexes,floor(size/len(holograms)),weights=weights))
    print(split_inds)
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
    
    mixed = mix([holo1,holo2,holo3,holo4,holo5])#,weights=[1,2,1,1,1])
    # print(mixed)
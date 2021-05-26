from collections import defaultdict
from config import config

def docname(i):
    return "___DOC"+str(i)+"___"

def count_co_occurences(doc, window_size, return_count=False, co_occurences=None):

    if not co_occurences:
        co_occurences = defaultdict(int)
    window_count = 0
    for i, w in enumerate(doc):
        for j in range(i + 1, min(i + window_size + 1, len(doc))):
            window_count += 1
            if (doc[i], doc[j]) in co_occurences:
                co_occurences[(doc[i], doc[j])] += 1 # Could add weighting based on distance
            else:
                co_occurences[(doc[j], doc[i])] += 1

    if return_count:
        return co_occurences, window_count
    return co_occurences
    
def create_idx_mapping(list, offset=0):
    index_mapping = {}

    for i, element in enumerate(list):
        index_mapping[element] = i + offset

    # print([str(a)+": "+str(b) for i, (a, b) in enumerate(index_mapping.items()) if i < 50])

    return index_mapping

# def create_idx_mapping(list):
#     index_mapping = {}

#     for element in list:
#         if element not in index_mapping:
#             index_mapping[element] = len(index_mapping)

#     print([str(a)+": "+str(b) for i, (a, b) in enumerate(index_mapping.items()) if i < 50])

#     return index_mapping
import fasttext

def evaluate_fasttext(docs, labels, tvt_idx, verbose=True):
    (train_idx, val_idx, test_idx, train_unlab_idx) = tvt_idx
    
    # make temp files with the train and test data since that is what FastText wants as input
    with open('other_models/eval.train.txt', mode='w') as train_file:
        for doc, label in zip(docs[train_idx], labels[train_idx]):
            doc_string = ''
            for word in doc:
                doc_string += word + ' '
            train_file.write('__label__' + str(label) + ' ' + doc_string[:-1] + '\n')
        for doc, label in zip(docs[val_idx], labels[val_idx]):
            doc_string = ''
            for word in doc:
                doc_string += word + ' '
            train_file.write('__label__' + str(label) + ' ' + doc_string[:-1] + '\n')
            
    with open('other_models/eval.test.txt', mode='w') as test_file:
        for doc, label in zip(docs[test_idx], labels[test_idx]):
            doc_string = ''
            for word in doc:
                doc_string += word + ' '
            test_file.write('__label__' + str(label) + ' ' + doc_string[:-1] + '\n')

    if verbose:
        print("[fasttext] training...")
    model = fasttext.train_supervised('other_models/eval.train.txt')

    if verbose:
        print("[fasttext] testing...")
    result = model.test('other_models/eval.test.txt')[1]
    
    if verbose:
        print("[fasttext] Got result %.4f" % result)

    return result

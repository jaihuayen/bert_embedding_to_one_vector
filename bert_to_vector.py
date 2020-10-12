def bert_to_vector(text_dataframe):
    
    import numpy as np
    import pandas as pd
    from bert_embedding import BertEmbedding
    
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    vectors = bert_embedding(text_dataframe.tolist(), 'avg')
    vectors_mean = [np.mean(k) for i in range(len(vectors)) for k in zip(*vectors[i][1])]
    vectors_mean_dataframe = pd.DataFrame(np.array(vectors_mean).reshape(len(vectors),-1))
    return vectors_mean_dataframe

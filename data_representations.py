from sklearn.feature_extraction import DictVectorizer
from collections import Counter, defaultdict
from math import ceil
import pandas as pd
import numpy as np


# ==========================================================================================================================================
# ==========================================================================================================================================
#                                                           BoW E N-GRAMMI
# ==========================================================================================================================================
# ==========================================================================================================================================

# Metodo che filtra le features in base alle occorrenze
# Usato per BoW, n-grammi

def filter_features(train_documents, test_documents, min_perc=0.0, max_perc=1.0):
    min_occ = len(train_documents) * min_perc
    max_occ = len(train_documents) * max_perc

    # i conteggi si fanno solo nel training set
    feature_counts = Counter()
    for doc in train_documents:
        for term in doc.features.keys():   # presenza nel documento
            feature_counts[term] += 1

    filtered_features = {feature for feature, count in feature_counts.items() if min_occ <= count <= max_occ}
    print(f'Rimosse feature che compaiono in meno di {ceil(min_occ)} documenti e in più di {ceil(max_occ)} documenti.\nSi passa da {len(list(feature_counts.keys()))} a {len(list(filtered_features))} features.')

    # applichiamo il filtro calcolato sul training set, sia al training set che al test set
    for document in train_documents:
        document.features = {feature: count for feature, count in document.features.items() if feature in filtered_features}
    
    for document in test_documents:
        document.features = {feature: count for feature, count in document.features.items() if feature in filtered_features}
    
    return train_documents, test_documents


# Metodo che crea la matrice delle feature "numerica" a partire da dizionari di conteggi
# Usato per BoW, n-grammi

def convert_dict_to_matrix(train_documents, test_documents):
    # creiamo la lista dei dizionari di features per il train e il test set
    train_features = [doc.features for doc in train_documents]
    test_features = [doc.features for doc in test_documents]
    
    # il dict vectorizer è calcolato solo sul training set e poi applicato sia al training che al test set
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_features)
    X_test = vectorizer.transform(test_features)
    
    feature_names = vectorizer.get_feature_names_out().tolist()
    return X_train, X_test, feature_names


# ==========================================================================================================================================
# ==========================================================================================================================================
#                                                                   N-GRAMMI
# ==========================================================================================================================================
# ==========================================================================================================================================


# Metodo che estrae gli ngrammi di parole

def extract_word_ngrams(document, item, n):
    if item == "word":
        all_words = document.get_words()
    elif item == "lemma":
        all_words = document.get_lemmas()
    elif item == "pos":
        all_words = document.get_pos_tags()
    else:
        raise ValueError(f"Invalid item: {item}")

    ngrams = defaultdict(int)
    for i in range(0, len(all_words) - n + 1):
        ngram_items = all_words[i : i + n]
        ngram = f"{item}_{n}_" + "_".join(ngram_items)
        ngrams[ngram] += 1

    return dict(ngrams)

# Metodo che estrae gli ngrammi di caratteri

def extract_char_ngrams(document, n):
    all_words = " ".join(document.get_words())

    ngrams = defaultdict(int)
    for i in range(0, len(all_words) - n + 1):
        ngram_chars = all_words[i : i + n]
        ngram = f"CHAR_{n}_" + ngram_chars
        ngrams[ngram] += 1

    return dict(ngrams)

# Metodo che normalizza i conteggi degli n-grammi in base alla lunghezza del documento
# Passa da conteggi a frequenze

def normalize_ngrams(ngrams, max_len):
    for ngram in ngrams:
        ngrams[ngram] = ngrams[ngram] / float(max_len)
    return ngrams


# metodo che estrae gli n-grammi di forme, lemmi, pos e caratteri da un documento

def extract_document_ngrams(document, word_ngrams=[], lemma_ngrams=[], pos_ngrams=[], char_ngrams=[]):
    num_tokens = max(document.num_tokens(), 1)
    num_chars = max(document.num_chars(), 1)

    document_ngrams = dict()

    for n in word_ngrams:
        ngrams = extract_word_ngrams(document, 'word', n)
        ngrams = normalize_ngrams(ngrams, num_tokens)
        document_ngrams.update(ngrams)

    for n in lemma_ngrams:
        ngrams = extract_word_ngrams(document, 'lemma', n)
        ngrams = normalize_ngrams(ngrams, num_tokens)
        document_ngrams.update(ngrams)

    for n in pos_ngrams:
        ngrams = extract_word_ngrams(document, 'pos', n)
        ngrams = normalize_ngrams(ngrams, num_tokens)
        document_ngrams.update(ngrams)

    for n in char_ngrams:
        ngrams = extract_char_ngrams(document, n)
        ngrams = normalize_ngrams(ngrams, num_chars)
        document_ngrams.update(ngrams)

    return document_ngrams

# ==========================================================================================================================================
# ==========================================================================================================================================
#                                                PROFILING-UD E EMBEDDINGS
# ==========================================================================================================================================
# ==========================================================================================================================================

# Metodo che converte la lista di feature dei documenti in una matrice numerica da dare in input al classificatore
def convert_list_to_matrix(train_documents, test_documents):
    train_features = [doc.features for doc in train_documents]
    test_features = [doc.features for doc in test_documents]
    
    X_train = np.array(train_features)
    X_test = np.array(test_features)
    
    return X_train, X_test


# ==========================================================================================================================================
# ==========================================================================================================================================
#                                                          EMBEDDINGS
# ==========================================================================================================================================
# ==========================================================================================================================================

# Metodo che carica gli embeddings da file e li mette in un dizionario
def load_word_embeddings(src_path):
    embeddings = dict()
    for line in open(src_path, 'r'):
        line = line.strip().split('\t')
        word = line[0]
        embedding = line[1:]
        embedding = [float(comp) for comp in embedding] # convertiamo le componenti dell'embedding in float
        embeddings[word] = np.asarray(embedding) # trasformiamo la lista delle componenti in un vettore di numpy
    return embeddings

# Metodo che calcola la media di embedding 
def compute_embeddings_mean(document_embeddings, embeddings_dim):
    if len(document_embeddings) > 0:
        return np.mean(document_embeddings, axis=0)
    else:
        return np.zeros(embeddings_dim)
    
# media di tutti gli embedding del documento

def compute_all_embeddings_mean(document, embeddings, embeddings_dim):
    document_embeddings = []
    
    for word in document.get_words():
        if word in embeddings:
            document_embeddings.append(embeddings[word])
    
    return compute_embeddings_mean(document_embeddings, embeddings_dim)


# media dei word embedding di parole piene

def compute_filtered_embeddings_mean(document, embeddings, embeddings_dim):
    document_embeddings = []
    
    for token in document.get_tokens():
        word = token.word
        pos = token.pos
        if word in embeddings and pos in ['ADJ', 'NOUN', 'VERB']:
            document_embeddings.append(embeddings[word])
    
    return compute_embeddings_mean(document_embeddings, embeddings_dim)


# media separata dei word embedding di parole piene

def compute_filtered_embeddings_sep_means(document, embeddings, embeddings_dim):
    target_pos = ['ADJ', 'NOUN', 'VERB']
    document_embeddings = {pos: [] for pos in target_pos}

    for token in document.get_tokens():
        word = token.word
        pos = token.pos
        if word in embeddings and pos in target_pos:
            document_embeddings[pos].append(embeddings[word])

    mean_document_embeddings = []
    for pos in target_pos:
        mean_document_embeddings.append(compute_embeddings_mean(document_embeddings[pos], embeddings_dim))

    mean_document_embeddings = np.concatenate(mean_document_embeddings, axis=0)
    return mean_document_embeddings


# ==========================================================================================================================================
# ==========================================================================================================================================
#                                                 METODI DI ESTRAZIONE DELLE FEATURES
# ==========================================================================================================================================
# ==========================================================================================================================================

# Metodo che crea la matrice delle features usitlizzando i BoW

def build_bow_features(train_documents, test_documents, min_perc=0.01, max_perc=0.2):
    # Estrazione delle parole dei documenti
    for document in train_documents:
        document.features = Counter(document.get_words())
    for document in test_documents:
        document.features = Counter(document.get_words())
    
    # Filtro delle parole troppo/troppo poco frequenti
    filter_features(train_documents, test_documents, min_perc, max_perc)

    # Creazione della matrice delle features
    X_train, X_test, features_names = convert_dict_to_matrix(train_documents, test_documents)
    
    return X_train, X_test, features_names

# Metodo che crea la matrice delle features usitlizzando gli n-grammi

def build_ngram_features(train_documents, test_documents, word_ngrams=[1, 2], lemma_ngrams=[], pos_ngrams=[], char_ngrams=[2], min_perc=0.1, max_perc=0.5):
    for document in train_documents:
        document.features = extract_document_ngrams(document, word_ngrams, lemma_ngrams, pos_ngrams, char_ngrams)
    for document in test_documents:
        document.features = extract_document_ngrams(document, word_ngrams, lemma_ngrams, pos_ngrams, char_ngrams)

    # Filtro delle parole troppo/troppo poco frequenti
    filter_features(train_documents, test_documents, min_perc, max_perc)

    # Creazione della matrice delle features
    X_train, X_test, features_names = convert_dict_to_matrix(train_documents, test_documents)
    
    return X_train, X_test, features_names


# Metodo che carica le features partendo dall'outptud di profiling-UD
def build_profiling_features(train_documents, test_documents, profiling_path):
    profiling_features = pd.read_csv(profiling_path, delimiter='\t')

    # Nomi delle features
    features_names = profiling_features.columns.tolist()
    features_names.remove('Filename')

    # Mapping dal nome del file al doc_id
    profiling_features = profiling_features.copy()
    profiling_features["doc_id"] = (profiling_features["Filename"].str.split("/").str[-1].str.replace(".conllu", ""))

    profiling_features.set_index("doc_id", inplace=True)
    profiling_features = profiling_features[features_names]

    for document in train_documents:
        document.features = profiling_features.loc[document.doc_id].tolist()
    for document in test_documents:
        document.features = profiling_features.loc[document.doc_id].tolist()

    # Creazione della matrice delle features
    X_train, X_test = convert_list_to_matrix(train_documents, test_documents)

    return X_train, X_test, features_names

# Metodo che carica le features partendo dagli embedding delle parole dei documenti
def build_embedding_features(train_documents, test_documents, embeddings_path, embeddings_dim, method="filtered_sep_means"):
    embeddings = load_word_embeddings(embeddings_path)

    for document in train_documents:
        if method == "all_mean":
            document.features = compute_all_embeddings_mean(document, embeddings, embeddings_dim)
        elif method == "filtered_mean":
            document.features = compute_filtered_embeddings_mean(document, embeddings, embeddings_dim)
        elif method == "filtered_sep_means":
            document.features = compute_filtered_embeddings_sep_means(document, embeddings, embeddings_dim)
        else:
            raise ValueError(f"Invalid method: {method}")
    for document in test_documents:
        if method == "all_mean":
            document.features = compute_all_embeddings_mean(document, embeddings, embeddings_dim)
        elif method == "filtered_mean":
            document.features = compute_filtered_embeddings_mean(document, embeddings, embeddings_dim)
        elif method == "filtered_sep_means":
            document.features = compute_filtered_embeddings_sep_means(document, embeddings, embeddings_dim)
        else:
            raise ValueError(f"Invalid method: {method}")

    X_train, X_test = convert_list_to_matrix(train_documents, test_documents)

    return X_train, X_test, list(range(embeddings_dim))

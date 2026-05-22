# Human vs AI Text Detection — Linguistica Computazionale 2

Progetto di classificazione automatica di testi in italiano per distinguere
testi scritti da **esseri umani** da testi generati da **intelligenze artificiali**.
Il task rientra nella shared task **DESEGMA-it (SubTask A)**.

Sono sviluppati due classificatori indipendenti basati su SVM, con feature
linguistiche di tipo diverso, costruiti a partire da annotazioni morfosintattiche
in formato CoNLL-U.

---

## Dataset

**Task:** classificazione binaria su testi in italiano
- Classe `0` — testo scritto da un essere umano
- Classe `1` — testo generato da un'intelligenza artificiale

**Fonte:** DESEGMA-it SubTask A (file CSV di training e test forniti dalla shared task)

**Composizione del dataset costruito:**

| Split | Human | AI | Totale |
|---|---|---|---|
| Training | 1000 | 1000 | 2000 |
| Validation | 500 | 500 | 1000 |
| Test | 500 | 500 | 1000 |
| **Totale** | **2000** | **2000** | **4000** |

Il dataset è bilanciato per classe. La suddivisione è fissa e riproducibile
(`random_state=26`). I documenti vengono identificati da un `doc_id` nel
formato `{split}_{indice}` e mappati su label e split in un file
`document_info.json`.

---

## Architettura del codice

### `classes.py` — strutture dati

**`Token`**
- Attributi: `word`, `lemma`, `pos`
- Metodi: `num_chars()`

**`Sentence`**
- Attributi: `text`, `tokens`
- Metodi: `add_token()`, `get_words()`, `get_lemmas()`, `get_pos_tags()`,
  `num_tokens()`, `num_chars()`

**`Document`**
- Attributi: `path`, `doc_id`, `split`, `label`, `sentences`, `features`
- Metodi: `get_tokens()`, `get_words()`, `get_lemmas()`, `get_pos_tags()`,
  `num_tokens()`, `num_chars()`, `num_sentences()`
- `load_sentences_from_conllu()`: parser del formato CoNLL-U — estrae word
  (col. 1), lemma (col. 2), POS (col. 3); gestisce token multi-parola,
  commenti e file senza riga vuota finale

---

### `data_representations.py` — estrazione delle feature

**N-grammi**
- `extract_word_ngrams(document, item, n)` — n-grammi di parole, lemmi o POS;
  frequenze normalizzate per lunghezza del documento
- `extract_char_ngrams(document, n)` — n-grammi di caratteri sull'intero testo
- `extract_document_ngrams(...)` — orchestratore che combina tutti i tipi di n-grammi

**Filtraggio e vettorizzazione**
- `filter_features(train, test, min_perc, max_perc)` — mantiene le feature
  presenti in una percentuale di documenti di training compresa tra `min_perc`
  e `max_perc`
- `convert_dict_to_matrix(train, test)` — converte i dizionari di feature
  in matrici sparse tramite `DictVectorizer` (fit solo su training)

**Profiling linguistico**
- `build_profiling_features(train, test, profiling_path)` — legge il CSV
  prodotto da Profiling-UD e restituisce array densi

**Word embeddings** (funzionalità disponibile)
- `load_word_embeddings(src_path)` — carica embedding da file TSV
- `compute_all_embeddings_mean` — media di tutti i token
- `compute_filtered_embeddings_mean` — media dei soli content word (ADJ, NOUN, VERB)
- `compute_filtered_embeddings_sep_means` — medie separate per ADJ, NOUN, VERB
  concatenate in un unico vettore

---

## Classificatori

### 1. Primo classificatore — SVM con N-grammi
**Notebook:** `primo_classificatore/classificatore_svm_ngrammi.ipynb`

Quattro configurazioni di feature testate tramite **5-fold cross-validation**
sul training set:

| Configurazione | Feature | F1 (5-fold CV) | Feature dopo filtraggio |
|---|---|---|---|
| Lessicale | Word 1-grammi + 2-grammi | **0.9730** | 7.387 |
| Morfologica | Char 2,3,4-grammi | 0.9680 | 17.192 |
| Mista | Lemma [1] + POS [2] + Char [3] | 0.9585 | 6.511 |
| Sintattica | POS 2,3-grammi | 0.8928 | 1.093 |

**Configurazione selezionata:** lessicale (word unigrams + bigrams)

**Pipeline del classificatore:**
1. Estrazione n-grammi normalizzati per lunghezza del documento
2. Filtraggio per frequenza: feature presenti nell'1%–20% dei documenti di training
3. Vettorizzazione sparsa con `DictVectorizer`
4. Scaling con `MaxAbsScaler` (mantiene la sparsità)
5. Addestramento `LinearSVC` (`dual=False`, `max_iter=5000`, `random_state=26`)
6. Valutazione con K-fold (k=5) su validation e test set
7. Confronto con baseline `DummyClassifier` (strategia `most_frequent`)

---

### 2. Secondo classificatore — SVM con Profiling linguistico
**Notebook:** `secondo_classificatore/classificatore_svm_profiling.ipynb`

Le feature sono estratte da **Profiling-UD**, uno strumento esterno che
analizza i file CoNLL-U e produce oltre 32 metriche linguistiche quantitative,
tra cui: distribuzione dei POS, statistiche morfologiche, relazioni di
dipendenza sintattica, pattern stilistici (punteggiatura, uso delle maiuscole).

**Pipeline del classificatore:**
1. Lettura del CSV prodotto da Profiling-UD
2. Allineamento filename → `doc_id`
3. Scaling con `StandardScaler` (media 0, varianza 1)
4. Addestramento `LinearSVC` (`dual=False`, `max_iter=5000`, `random_state=26`)
5. Valutazione con K-fold (k=5) su validation e test set
6. Analisi dei coefficienti: top 10 feature positive e negative
7. Analisi dei casi incerti: documenti con decision score |s| < 0.5

---

## Valutazione

**Metrica principale:** F1 macro-averaged

**Metodi di valutazione applicati a entrambi i classificatori:**
- K-fold cross-validation (k=5) sul training set
- Valutazione per fold su validation e test set
- Confronto con baseline `DummyClassifier (most_frequent)`
- Report di classificazione (precision, recall, F1 per classe)
- Matrice di confusione

**Analisi aggiuntive nel secondo classificatore:**
- Visualizzazione dei coefficienti del modello (feature più discriminative)
- Analisi dei casi borderline (score vicini allo zero con testi di esempio)

---

## Classificatori da implementare

### 3. SVM + Word Embeddings `[Da implementare]`

Rappresentazione vettoriale del testo basata su word embedding pre-addestrati
(es. modelli Italian Word Embeddings di ItaliaNLP Lab). Sono previste diverse
strategie di combinazione degli embedding delle singole parole da confrontare
sul validation set:
- Media di tutti i token del documento
- Media dei soli content word (ADJ, NOUN, VERB)
- Medie separate per categoria grammaticale, concatenate

Il sistema migliore sul validation set verrà poi valutato sul test set.
L'infrastruttura è già predisposta in `data_representations.py`.

### 4. Fine-tuning di un Neural Language Model `[Da implementare]`

Fine-tuning per 3 epoche di un neural language model pre-addestrato (es. BERT
multilingue o un modello italiano). Sono richiesti:
- Curve di loss su training e validation set per ogni epoca
- Prestazioni sul validation set al termine di ciascuna epoca
- Valutazione finale sul test set al termine della terza epoca

---

## Dipendenze

pandas
numpy
scikit-learn
matplotlib



**Tool esterno richiesto:**
- **Profiling-UD** — produce le annotazioni CoNLL-U e il CSV di feature
  linguistiche usato dal secondo classificatore

---

## Ordine di esecuzione

1. `dataset/built_dataset.ipynb` — crea il dataset bilanciato e i file di testo
2. Esecuzione di **Profiling-UD** sulla directory `profiling_input/` →
   genera `profiling_output/` (CoNLL-U) e `linguistic_profiling.csv`
3. `primo_classificatore/classificatore_svm_ngrammi.ipynb`
4. `secondo_classificatore/classificatore_svm_profiling.ipynb`

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier

# Creamos nuestras opiones sobre la película.
opinion1 = "Visceral, stunning and relentless film making. Dicaprio's Herculean, almost purely physical performance" \
            "and Hardy's wide eyed intensity coupled with the almost overwhelming beauty of the landscape - those " \
            "trees, the natural light, the sun peeking through the clouds, rendered the proceedings down to savage" \
            "poetry. A hypnotic, beautiful, exhausting film."

opinion2 = "I saw this film on Friday. For the first 40 minutes involving spoken dialogue they need not have " \
            "bothered. For me the dialogue was totally unintelligible with grunting, southern states drawl, " \
            "and coarse accent that made it impossible to understand what they were saying."

opinion3 = "It was a idiotic film that produces a magnificent fascination."

# Método que transforma las palabras del texto a un diccionario.
# Es importante para que el que clasificador funcione ya que solo acepta conjuntos de características.
def word_feats(words):
    return dict([(word,True) for word in words])

# Seleccionamos los textos negativos y positivos del conjunto de entrenamiento.
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

# Generamos el conjunto de diccionarios con las palabras de cada uno de los textos de entrenamiento.
negfeats = [(word_feats(movie_reviews.words(fileids=[file])),'neg') for file in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[file])),'pos') for file in posids]

# Seleccionamos 3/4 como entrenamiento y el resto como test.
negcutoff = int(len(negfeats) * 3 / 4)
poscutoff = int(len(posfeats) * 3 / 4)

# Generamos los conjuntos sobre los diccionarios positivos y negativos.
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print("Instancias de test: %d sobre: %d instancias de entrenamiento. " % (len(testfeats),len(trainfeats)))

# Entrenamos al clasificador con los conjuntos de entrenamiento.
classifier = NaiveBayesClassifier.train(trainfeats)

# Obtenemos la precisión del clasificador utilizando el conjunto de test.
print("Precisión:", nltk.classify.util.accuracy(classifier,testfeats))

# Mostramos las características más importantes identificadas por el clasificador.
# Se muestra la influencia de las palabras más importantes para realizar la clasificación positiva o negativa.
classifier.show_most_informative_features()

# EJ. 4
def test_opinion(word_tokenized):
    for s in word_tokenized:
        for w in s:
            

word_tokenized1 = [nltk.tokenize.word_tokenize(s) for s in nltk.tokenize.PunktSentenceTokenizer().tokenize(opinion1)]
word_tokenized2 = [nltk.tokenize.word_tokenize(s) for s in nltk.tokenize.PunktSentenceTokenizer().tokenize(opinion2)]
word_tokenized3 = [nltk.tokenize.word_tokenize(s) for s in nltk.tokenize.PunktSentenceTokenizer().tokenize(opinion3)]


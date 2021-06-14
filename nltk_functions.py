
#what we need for NLP
import nltk
import spacy
from spacy_lefff import LefffLemmatizer
from spacy.language import Language
import unidecode

def remove_accents(text):
    unaccented_string = unidecode.unidecode(text)
    return unaccented_string

@Language.factory('french_lemmatizer')
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer()

nlp = spacy.load('fr_core_news_sm')
nlp.add_pipe('french_lemmatizer', name='lefff')

def lemmatize(sentence):
    dc = nlp(sentence)
    L=[]
    for d in dc:
        mot = d.lemma_.lower()
        mot = remove_accents(mot)
        L.append(mot)
    return L


 #split sentence into array of words/tokens a token can be a word or punctuation character, or number
def tokenize(phrase):
    return nltk.word_tokenize(phrase)

def bag_of_words(phrase,all_words):
    bag = []
    phrase2 = []
    for i in phrase : phrase2.extend(lemmatize(i))
    for m in all_words:
        if m in phrase2:
            bag.append(1)
        else: bag.append(0)
    return bag

#chatbot functions
def clean_up_sentence(sentence):
    sentence_words = tokenize(sentence)
    l = []
    for m in sentence_words:
        l.extend(lemmatize(m))
    return l

print(lemmatize("fête éliminer père"))




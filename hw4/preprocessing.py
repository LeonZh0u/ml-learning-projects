import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS

def preprocess_with_spacy(text: str):
    nlp = spacy.load('en')
    tokenized = [token.text for token in nlp(text)]
    lemmantized = []
    for token in nlp(text):
        # skip stopwords
        if not token in STOP_WORDS:
            print(token.text, token.lemma_)
            lemmantized.append([token.text, token.lemma_])

    return lemmantized

if __name__ == "__main__":
    text= "This is one of the greatest films ever made. Brilliant acting by George C. Scott and Diane Riggs. This movie is both disturbing and extremely deep. Don't be fooled into believing this is just a comedy. It is a brilliant satire about the medical profession. It is not a pretty picture. Healthy patients are killed by incompetent surgeons, who spend all their time making money outside the hospital. And yet, you really believe that this is a hospital. The producers were very careful to include real medical terminology and real medical cases. This movie really reveals how difficult in is to run a hospital, and how badly things already were in 1971. I loved this movie."
    print(preprocess_with_spacy(text))
    #print(preprocess_with_nltk(text))

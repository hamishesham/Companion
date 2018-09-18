import spacy
from spacy.pipeline import TextCategorizer


nlp = spacy.load('en')
nlp.disable_pipes('tagger', 'parser')
doc = nlp(u"this is a sentence")
textcat = TextCategorizer(nlp.vocab)
print(textcat(doc))
	
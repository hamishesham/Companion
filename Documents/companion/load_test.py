#load test
import spacy


TEST_DATA = [
     ("we are meeting eleven fourty a m today", {'entities': [(15, 29, 'TIME'), (31, 35, 'DATE')]}),
     ("the boys will meet three fifty a m tomorrow", {'entities': [(16, 30, 'TIME'), (33, 36, 'DATE')]}),
	 ("they will be there by one twenty a m", {'entities': [(22, 34, 'TIME')]})
]


nlp = spacy.load('/model')
for text, annotations in TEST_DATA:
	print('after training:')
	doc = nlp(text)
	print(doc)
	for entity in doc.ents:
		print(entity.text, entity.label_)
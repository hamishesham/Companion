# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
import random
import output
import text_to_spacy

#simple training example
#model = [sentence, {'entities':[(start_char_num, end_char_num, label)]}]
text_to_spacy.text2spacyFormat("text.txt", "output.py", "TRAIN_DATA")
TRAIN_DATA = output.TRAIN_DATA

TEST_DATA = [
     ("we are meeting eleven fourty a m today", {'entities': [(15, 31, 'TIME'), (31, 35, 'DATE')]}),
     ("the boys will meet three fifty a m tomorrow", {'entities': [(16, 30, 'TIME'), (33, 36, 'DATE')]}),
	 ("they will be there by one twenty a m", {'entities': [(22, 34, 'TIME')]})
]


nlp = spacy.load('en')
with nlp.disable_pipes('tagger', 'parser'):
    optimizer = nlp.begin_training()
for i in range(10):
	random.shuffle(TRAIN_DATA)
	for text, annotations in TRAIN_DATA:
		if i == 0:
			print('before training:')
			doc = nlp(text)
			print(doc)
			for entity in doc.ents:
				print(entity.text, entity.label_)
		nlp.update([text], [annotations], sgd=optimizer)

for text, annotations in TEST_DATA:
	print('after training:')
	doc = nlp(text)
	print(doc)
	for entity in doc.ents:
		print(entity.text, entity.label_)

nlp.to_disk('/model')
#the input file is a tap-separated sentences and the output is a spacy format:
#dataset = [(sentence, {'entities':[(start_char_index, end_char_index + 1, label),]}),]

import re


def get_index(sentence, key):
    index = 0
    if key in sentence:
        char = key[0]
        for ch in sentence:
            if ch == char:
                if sentence[index:index+len(key)] == key:
                    return index, index + len(key) - 1
            index += 1
    return -1

def text2spacyFormat(inputfilepath, outputfilepath, data_type):
	try:
		inputFile = open(inputfilepath, "r")
	except:
		raise Exception("the input file does not exist.")
	try:
		outputFile = open(outputfilepath, "w")
	except:
		raise Exception("may be you do not have the permission to write here.")
	counter = 0

	data = re.split('\n', inputFile.read())
	content = data_type + " = [\n"
	for line in data:
		counter = counter + 1
		word = re.split("\t", line)
		sentence = word[0]
		record = "\t('" + sentence + "', {'entities':["
		entity_count = int(word[1])
		for i in range(0, entity_count):
			key = word[(i*2 + 3 )]
			index = get_index(sentence, key)
			try :
				start_index, last_index = get_index(sentence, key)
			except :
				raise Exception("Sentence number " + str(counter) + " has a missmatch between an entity and the main sentence")
			label = "'" + word[(i + 1 )*2] + "'"
			record =  record + "(" + str(start_index) + ", " + str(last_index + 1) + ", " + label + "),"

		record = record + "]}),\n"
		content =content + record

	content = content + "]"
	outputFile.write(content)
	outputFile.close()
	return 	

if __name__ == '__main__':
	text2spacyFormat("text.txt", "output.py", "dataset")
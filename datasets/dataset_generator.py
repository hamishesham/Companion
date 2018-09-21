import re


inputFile = open("dataset.txt", "r")
outputFile = open("dataset.tsv", "w")

data = re.split('\n', inputFile.read())
length = len(data)
counter = length
for line in data:
	print("remaining : " + counter + " of : " + length)
	print("owner : " + line)
	outputFile.write("owner\t" + line + "\t" + input() + "\n")
	print("other : " + line)
	outputFile.write("other\t" + line + "\t" + input() + "\n")
	counter--

outputFile.close()
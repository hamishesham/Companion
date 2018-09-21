inputFile = open("dataset.text", "r")
outputFile = open("dataset.tsv", "w")

for line in inputFile:
	print("owner : " + line)
	outputFile.write("owner\t" + line + "\t" + input() + "\n")
	print("other : " + line)
	outputFile.write("other\t" + line + "\t" + input() + "\n")
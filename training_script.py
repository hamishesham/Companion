#training script
from os import system as call


call("\
		git clone https://github.com/UniversalDependencies/UD_Spanish-AnCora\
		mkdir ancora-json\
		python -m spacy convert UD_Spanish-AnCora/es_ancora-ud-train.conllu ancora-json\
		python -m spacy convert UD_Spanish-AnCora/es_ancora-ud-dev.conllu ancora-json\
		mkdir models\
		python -m spacy train es models ancora-json/es_ancora-ud-train.json ancora-json/es_ancora-ud-dev.json\
")
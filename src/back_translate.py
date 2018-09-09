# Before running this code, you need to set google application credentials in order to use Google Translate API:
# (See https://cloud.google.com/translate/docs/quickstart?csw=1&authuser=1 and follow the steps in 'Before You Begin'
# After step 2, you could run this code in your terminal session.

from google.cloud import translate
from google.auth import compute_engine
from google.cloud import storage

# Translate a text using Google Translate API
def translate_text(text, lang):
	translate_client = translate.Client()
	result = translate_client.translate(text, target_language = lang)
	return result['translatedText']

# Back translate a file with intermediate languages
def back_translate(file, inter_langs):
	source = open(file + '.txt', 'r')
	inter = open(file + '_inter.txt', 'w')
	back = open(file + '_back.txt', 'w')
	for line in source:
		cols = line.split('\t')
		for lang in inter_langs:
			# Translate to intermediate lang
			inter_result = translate_text(cols[1], lang)
			inter.write(cols[0] + '\t' + inter_result + '\t' + cols[2] + '\n')
			# Translate back to intermediate Chinese
			back_result = translate_text(inter_result, 'zh')
			back.write(cols[0] + '\t' + back_result + '\t' + cols[2] + '\n')

# Deduplicate translations in back translated file
def deduplicate_translations(back_name, num_lang):
	back_whole = open(back_name, 'r')
	back_dedup = open(back_name.rstrip('.txt') + '_dedup.txt', 'w')
	lines = back_whole.readlines()

	# Number of sentences in original data
	num_sent = int(len(lines)/num_lang)

	# Check if there are duplicate translations
	for i in range(num_sent):
		translations = []
		for id in range(num_lang):
			translation = lines[i*num_lang+id].split('\t')[1]
			if translation not in translations: # Retain unique translations only
				translations.append(translation)
				back_dedup.write(lines[i*num_lang+id])

def filter_label(back_dedup_name, labels_to_be_filtered):
	back_dedup = open(back_dedup_name, 'r')
	back_filtered = open(back_dedup_name.rstrip('dedup.txt')+'filter'+''.join(labels_to_be_filtered)+'.txt', 'w')
	for line in back_dedup:
		if line.strip().split('\t')[2] not in labels_to_be_filtered:
			back_filtered.write(line)

# data_path = '../data/'
# data_file_names = ['emo_train', 'verb_train']
# inter_langs = ['en', 'fr', 'ru', 'es', 'ja', 'ar'] # Intermediate languages: English, French, Russian, Spanish, Japanese, Arabic
# labels_to_be_filtered = [] # Labels that already have many sentences in original file might not need back translation; here we simply don't filter any label
#
# for name in data_file_names:
# 	file = data_path + name
#
# 	# Back translation
# 	back_translate(file, inter_langs)
#
# 	# Deduplicating
# 	back_name = file + '_back.txt'
# 	deduplicate_translations(back_name, len(inter_langs))
#
# 	# Filtering
# 	back_dedup_name = file + '_back_dedup.txt'
# 	filter_label(back_dedup_name, labels_to_be_filtered)
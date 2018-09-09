import xml.etree.ElementTree as etree

# Convert xml format to txt format, and convert int label to text label
def xml_to_txt(file_name, verb_list, emo_list):
	inf = open(file_name, 'r')
	ouf = open(file_name.rstrip('xml')+'txt', 'w')
	for action, elem in etree.iterparse(inf, events=['end']):
		if elem.tag == 'ID':
			ouf.write(elem.text+'\t')
		elif elem.tag == 'Sentence':
			ouf.write(elem.text)
		elif elem.tag == 'Label':
			ouf.write('\t'+verb_list[int(elem.text)]) # Convert int label to text label
		elif elem.tag == 'Emo_Class':
			ouf.write('\t'+emo_list[int(elem.text)-1]) # Convert int label to text label
		elif elem.tag == 'metaphor':
			ouf.write('\n')
		elem.clear()

# verb_list = ['负例', '动词', '名词']
# emo_list = ['乐', '好', '怒', '哀', '惧', '恶','惊']
# data_path = '../data/new_data/'
# data_file_names = ['隐喻动词_train.xml', '隐喻情感_train.xml']
# for name in data_file_names:
# 	xml_to_txt(data_path + name, verb_list, emo_list)


import json
import pandas as pd


def iob2(tags):
    """
    Check that tags have a valid BIO format.
    Tags in BIO1 format are converted to BIO2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def iob_iobes(tags):
    """
    the function is used to convert
    BIO -> BIOES tagging
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to BIO2
    Only BIO1 and BIO2 schemes are accepted for input data.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the BIO format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in BIO format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'BIOES':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Wrong tagging scheme!')

def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

# f = open("wnut17train.conll.txt", "r")

# f = open("/Users/satadisha/Documents/GitHub/conll_train.txt", "r")

# file_text=f.read()

# df_holder=[]
# df_columns=('Sentence #','Word','Tag')

# # sentences=file_text.split('\n\t\n') #wnut
# docs=list(filter(lambda elem: elem!='', file_text.split('-DOCSTART- -X- -X- O')))

# sentID=0

# for doc in docs:
# 	if(doc):
# 		sentences=doc.split('\n\n') #conll
# 		for sentence in sentences:
# 			token_level_entries=sentence.split('\n')
# 			for token_level_entry in token_level_entries:
# 				# print(token_level_entry)
# 				if(token_level_entry):
# 					# tabbed_entries=token_level_entry.split('\t')
# 					tabbed_entries=token_level_entry.split(' ')
# 					print(tabbed_entries)
# 					if not((tabbed_entries[0]=='')&(tabbed_entries[3]=='')):
# 						tag=tabbed_entries[3].split('-')[0]
# 						df_dict={'Sentence #':str(sentID), 'Word':tabbed_entries[0], 'Tag':tag}
# 						print(df_dict)
# 						df_holder.append(df_dict)
# 			sentID+=1

# f= open("/Users/satadisha/Documents/GitHub/Neuroner_Twilight/data/conll2003/en/test.txt", "r")
# file_text=f.read()

# docs=list(filter(lambda elem: elem!='', file_text.split('-DOCSTART- -X- -X- O')))
# writestuff=''
# for doc in docs:
# 	if(doc):
# 		sentences=doc.split('\n\n') #conll
# 		for sentence in sentences:
# 			token_level_entries=sentence.split('\n')
# 			for token_level_entry in token_level_entries:
# 				# print(token_level_entry)
# 				if(token_level_entry):
# 					# tabbed_entries=token_level_entry.split('\t')
# 					tabbed_entries=token_level_entry.split(' ')
# 					print(tabbed_entries)
# 					if not((tabbed_entries[0]=='')&(tabbed_entries[3]=='')):
# 						writestuff+=tabbed_entries[0]+'\t'+tabbed_entries[3]+'\n'
# 			writestuff+='\n'


f= open("/Users/satadisha/Documents/GitHub/emerging.dev.conll.preproc.url", "r")
file_write_text=''
file_text=f.read()
lines=file_text.split('\n')
for line in lines:
	to_write=''
	if(len(line)>0):
		tabs=line.split('\t')
		tag=tabs[1].split('-')[0]
		to_write=tabs[0]+'\t'+tag
	print(to_write)
	file_write_text+=to_write+'\n'
	# print(line)
	# print(len(line))
	print('=====')

f1= open("/Users/satadisha/Documents/GitHub/emerging.dev.conll.preproc.url.updated", "w")
f1.write(file_write_text)
f1.close()

# f = open("/Users/satadisha/Documents/GitHub/conlltest_BIO.txt", "w")

# f.write(writestuff)
# f.close()


# sentences=file_text.split('\n\n') #conll
# print(len(sentences))
# # print(sentences[0])

# for ind, sentence in enumerate(sentences):
# 	token_level_entries=sentence.split('\n')
# 	for token_level_entry in token_level_entries:
# 		# print(token_level_entry)
# 		if(token_level_entry):
# 			# tabbed_entries=token_level_entry.split('\t')
# 			tabbed_entries=token_level_entry.split(' ')
# 			print(tabbed_entries)
# 			if not((tabbed_entries[0]=='')&(tabbed_entries[3]=='')):
# 				tag=tabbed_entries[3].split('-')[0]
# 				df_dict={'Sentence #':str(ind), 'Word':tabbed_entries[0], 'Tag':tag}
# 				print(df_dict)
# 				df_holder.append(df_dict)



# df_out = pd.DataFrame(df_holder,columns=df_columns)
# print(len(df_out))

# print(df_out.head(30))

# df_out.to_csv("/Users/satadisha/Documents/GitHub/wnut17train_BIO.csv", sep=',', encoding='utf-8',index=False)
# df_out.to_csv("/Users/satadisha/Documents/GitHub/conlltrain_BIO.csv", sep=',', encoding='utf-8',index=False)


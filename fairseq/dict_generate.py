from fairseq.data import Dictionary
from fairseq import tokenizer

d = Dictionary()
Dictionary.add_file_to_dictionary('len_label.txt', d, tokenizer.tokenize_line, 1)
d.finalize(threshold=-1, nwords=-1, padding_factor=8)
d.save('./dict.txt')
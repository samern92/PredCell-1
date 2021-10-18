# Text_Loader.py

import io
import numpy as np

def get_onehot_details(text):
	chars=sorted(list(set(text)))
	n_chars = len(chars)
	char_to_ind_dict = {c: i for i, c in enumerate(chars)}
	ind_to_char_dict = {i: c for i, c in enumerate(chars)}
	return {"n_chars": n_chars, "char_to_ind_dict": char_to_ind_dict, "ind_to_char_dict": ind_to_char_dict}

class Text_Loader():
	def __init__(self, text, maxlen, step, onehot_details = None):
		self.segmented_data = None
		self.onehot_data = None
		self.data_length = None
		self.char_to_ind_dict = None
		self.ind_to_char_dict = None
		self.n_chars = None

		if onehot_details is None:
			onehot_details = get_onehot_details(text)
		self.char_to_ind_dict = onehot_details["char_to_ind_dict"]
		self.ind_to_char_dict = onehot_details["ind_to_char_dict"]
		self.n_chars = onehot_details["n_chars"]

		self.segmented_data, self.onehot_data = self.segment_data(text, maxlen, step)
		self.data_length = len(self.segmented_data)

	def sentences_to_indices_arr(self, sentences, maxlen):
		x = np.zeros((len(sentences), maxlen), dtype=np.int64)
		for i, sentence in enumerate(sentences):
			for t, char in enumerate(sentence):
				x[i, t] = self.char_to_ind_dict[char]
		return x

	def to_onehot(self, data, n_items=None):
		'''data: ndarray of integers beginning with 0'''
		n_items = data.max() + 1 if n_items is None else n_items
		ret = np.zeros(data.shape + (n_items,), dtype=np.float32)
		it = np.nditer(data, flags=['multi_index'])
		for val in it:
			ret[it.multi_index + (val,)] = 1
		return ret

	def segment_data(self, text, maxlen, step):
		# cut the text in semi-redundant sequences of maxlen characters
		step = 5
		sentences = []
		for i in range(0, len(text) - maxlen, step):
			sentences.append(text[i: i + maxlen])
		return np.array(sentences), self.to_onehot(self.sentences_to_indices_arr(sentences, maxlen), n_items = self.n_chars)

	def __len__(self):
		return self.data_length

	def __getitem__(self, idx):
		return self.onehot_data[idx]
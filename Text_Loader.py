# Text_Loader.py

import numpy as np
import torch

def split_text(text, proportion = [4,1]):
	onehot_details = get_onehot_details(text)
	# text = np.random.shuffle(text)
	split = int(len(text) * proportion[0] / sum(proportion))
	return text[:split], text[split:], onehot_details

def get_onehot_details(text):
	chars=sorted(list(set(text)))
	n_chars = len(chars)
	char_to_ind_dict = {c: i for i, c in enumerate(chars)}
	ind_to_char_dict = {i: c for i, c in enumerate(chars)}
	return {"n_chars": n_chars, "char_to_ind_dict": char_to_ind_dict, "ind_to_char_dict": ind_to_char_dict}

def onehot_to_char(onehot_encodings, ind_to_char_dict, return_string = False):
	chars = np.argmax(onehot_encodings, axis = 1)
	results = []
	for c in chars: results.append(ind_to_char_dict[int(c)])
	if return_string == True:
		results = "".join(results)
	return results

class Text_Loader():
	def __init__(self, text, maxlen, step, onehot_details = None, override_initialization = False):
		self.segmented_data = None
		self.onehot_data = None
		self.data_length = None
		self.char_to_ind_dict = None
		self.ind_to_char_dict = None
		self.n_chars = None

		if override_initialization == False:
			if onehot_details is None:
				onehot_details = get_onehot_details(text)
			self.char_to_ind_dict = onehot_details["char_to_ind_dict"]
			self.ind_to_char_dict = onehot_details["ind_to_char_dict"]
			self.n_chars = onehot_details["n_chars"]

			self.segmented_data, self.onehot_data = self.segment_data(text, maxlen, step)
			self.data_length = len(self.segmented_data)

	def segment_data(self, text, maxlen, step):
		# cut the text in semi-redundant sequences of maxlen characters
		step = 5
		sentences = []
		for i in range(0, len(text) - maxlen, step):
			sentences.append(text[i: i + maxlen])
		return np.array(sentences), self.__to_onehot(self.__sentences_to_indices_arr(sentences, maxlen), n_items = self.n_chars)

	def split_data(self, proportion = []):
		inds = np.arange(self.data_length, dtype = int)
		np.random.shuffle(inds)
		new_instances = []
		running_len = self.data_length
		curr_start = 0
		for por_ind in range(len(proportion)):
			curr_end = int(self.data_length/sum(proportion[por_ind:])*proportion[por_ind])
			new_inst = Text_Loader(None, None, None, override_initialization = True)
			curr_inds = inds[curr_start:curr_end]
			new_inst.segmented_data = self.segmented_data[curr_inds].copy()
			new_inst.onehot_data = self.onehot_data[curr_inds].copy()
			new_inst.data_length = len(new_inst.segmented_data)
			new_inst.char_to_ind_dict = self.char_to_ind_dict.copy()
			new_inst.ind_to_char_dict = self.ind_to_char_dict.copy()
			new_inst.n_chars = self.n_chars
			new_instances.append(new_inst)
			curr_start = curr_end
			running_len -= new_inst.data_length
		return new_instances

	def reconstruct_chars(self, onehot_encoding, return_string = False):
		chars = np.argmax(onehot_encoding, axis = 1)
		# chars = torch.argmax(onehot_encoding, axis = 1)
		results = []
		for c in chars: results.append(self.ind_to_char_dict[int(c)])
		if return_string == True:
			results = "".join(results)
		return results

	def __len__(self):
		return self.data_length

	def __getitem__(self, idx):
		return self.onehot_data[idx]

#							  Private Functions  							   #

	def __sentences_to_indices_arr(self, sentences, maxlen):
		x = np.zeros((len(sentences), maxlen), dtype=np.int64)
		for i, sentence in enumerate(sentences):
			for t, char in enumerate(sentence):
				x[i, t] = self.char_to_ind_dict[char]
		return x

	def __to_onehot(self, data, n_items=None):
		'''data: ndarray of integers beginning with 0'''
		n_items = data.max() + 1 if n_items is None else n_items
		ret = np.zeros(data.shape + (n_items,), dtype=np.float32)
		it = np.nditer(data, flags=['multi_index'])
		for val in it:
			ret[it.multi_index + (val,)] = 1
		return ret

# predcell_train_new.py

import io
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


from Text_Loader import *
from predcell_subtractive_relu import *

TEXT_FILE = "./nietzsche.txt"
MAXLEN = len("the quick brown fox jumps over the lazy dog ")
STEP = 5
TRAIN_SIZE = 50000
TEST_SIZE = 2000
DL_PARAMS = {"batch_size": 32, "shuffle": True}
NUM_LSTEM = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("Using " + str(torch.cuda.device_count()) + " devices") 

def main():
	text = get_text(TEXT_FILE)

	train_text = (text)[:TRAIN_SIZE]
	FNT_train = Text_Loader(train_text, MAXLEN, STEP)
	train_dl = DataLoader(FNT_train, **DL_PARAMS)

	test_text = (text)[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
	FNT_test = Text_Loader(test_text, MAXLEN, STEP)
	test_dl = DataLoader(FNT_test, **DL_PARAMS)

	print (FNT_train.n_chars)
	predcell = PredCells(NUM_LSTEM + 1, MAXLEN, 128, FNT_train.n_chars)
	trainable_st_params = [p for model in predcell.st_units for p in model.parameters() if p.requires_grad]
	trainable_err_params = [p for model in predcell.err_units for p in model.parameters() if p.requires_grad]
	# Get all the parameters along with their associated names.
	names_and_params = []
	for lyr, (st_unit, err_unit) in enumerate(zip(predcell.st_units, predcell.err_units)):
		names_and_params.append((f'st_units[{lyr}].V.weight', st_unit.V.weight))
		names_and_params.append((f'st_units[{lyr}].V.bias', st_unit.V.bias))

		if type(st_unit.LSTM_) is torch.nn.modules.rnn.LSTM:
			names_and_params.append((f'st_units[{lyr}].LSTM.weight_ih_l', st_unit.LSTM_.weight_ih_l0))
			names_and_params.append((f'st_units[{lyr}].LSTM.weight_hh_l', st_unit.LSTM_.weight_hh_l0))
			names_and_params.append((f'st_units[{lyr}].LSTM.bias_ih_l', st_unit.LSTM_.bias_ih_l0))
			names_and_params.append((f'st_units[{lyr}].LSTM.bias_hh_l', st_unit.LSTM_.bias_hh_l0))
		elif type(st_unit.LSTM_) is torch.nn.modules.rnn.LSTMCell:
			names_and_params.append((f'st_units[{lyr}].LSTM.weight_ih', st_unit.LSTM_.weight_ih))
			names_and_params.append((f'st_units[{lyr}].LSTM.weight_hh', st_unit.LSTM_.weight_hh))
			names_and_params.append((f'st_units[{lyr}].LSTM.bias_ih', st_unit.LSTM_.bias_ih))
			names_and_params.append((f'st_units[{lyr}].LSTM.bias_hh', st_unit.LSTM_.bias_hh))
		names_and_params.append((f'err_units[{lyr}].W.weight', err_unit.W.weight))
		names_and_params.append((f'err_units[{lyr}].W.bias', err_unit.W.bias))
	trainable_params = trainable_st_params + trainable_err_params

	# predcell = torch.nn.DataParallel(predcell)
	predcell = predcell.to(device)
	optimizer = torch.optim.Adam(trainable_params, lr=8e-4)

	train_losses = []
	first_layer_train_losses = []
	for batch_id, sentences in enumerate(tqdm(train_dl, total=len(train_dl), disable = True)):
		print (sentences.shape)
		print ("batch_id:" + str(batch_id))
		curr_sent = sentences.to(torch.device(device))
		print("Outside: input size", curr_sent.size())
		predcell.init_vars()
		loss, first_layer_loss, predictions = predcell.forward(curr_sent)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		train_losses.append(loss.detach().item())
		first_layer_train_losses.append(first_layer_loss.detach().item())

	return

def get_text(file_name):
	path = file_name
	with io.open(path, encoding="utf-8") as f:
		text = f.read().lower()
	text = text.replace("\n", " ")
	return text

if __name__ == "__main__":
	main()
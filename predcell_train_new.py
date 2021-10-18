# predcell_train_new.py

import io
import sys
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Text_Loader import *
from predcell_subtractive_relu import *

TEXT_FILE = "./nietzsche.txt"
MAXLEN = len("the quick brown fox jumps over the lazy dog ")
STEP = 5
TRAIN_TEST_PORP = [5,2]
DL_PARAMS = {"batch_size": 32, "shuffle": True}
NUM_LSTEM = 1
NUM_EPOCHS = 200

# device = torch.device("cuda" if torch.cuda.i_available() else "cpu")
device = torch.device("cpu")
print ("Using " + str(torch.cuda.device_count()) + " devices") 

def main():

# Load dataset
################################################################################

	text = get_text(TEXT_FILE)[:200]
	onehot_details = get_onehot_details(text)
	text_len = len(text)
	tt_mark = int(text_len*TRAIN_TEST_PORP[0]/(TRAIN_TEST_PORP[0]+TRAIN_TEST_PORP[1]))
	print ("Training size: " + str(tt_mark))
	print ("Testing size: " + str(text_len - tt_mark))

	train_text = (text)[:tt_mark]
	FNT_train = Text_Loader(train_text, MAXLEN, STEP, onehot_details = onehot_details)
	train_dl = DataLoader(FNT_train, **DL_PARAMS)

	test_text = (text)[tt_mark:]
	FNT_test = Text_Loader(test_text, MAXLEN, STEP, onehot_details = onehot_details)
	test_dl = DataLoader(FNT_test, **DL_PARAMS)

# Define Model
################################################################################

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

# Training Model
################################################################################

	predcell = predcell.to(device)
	optimizer = torch.optim.Adam(trainable_params, lr=8e-4)

	all_train_loss = []
	all_test_loss = []
	# for epoch in trange(NUM_EPOCHS):
	for epoch in range(NUM_EPOCHS):
		print ("\nEpoch: " + str(epoch) + "\n################################################################################\n")
		
		# train
		print ("Start Training")
		train_losses = []
		first_layer_train_losses = []
		for batch_id, sentences in enumerate(tqdm(train_dl, total=len(train_dl), disable = True)):
			# print ("batch_id:" + str(batch_id))
			curr_sent = sentences.to(torch.device(device))
			# print("Outside: input size", curr_sent.size())
			predcell.init_vars(batch_size = sentences.shape[0])
			loss, first_layer_loss, predictions = predcell.forward(curr_sent)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			train_losses.append(loss.detach())
			first_layer_train_losses.append(torch.mean(first_layer_loss.detach()).numpy())
		mean_train_losses = np.mean(train_losses)
		mean_first_layer_train_losses = np.mean(first_layer_train_losses)
		all_train_loss.append(mean_train_losses)

		# validation
		print ("Start Validating")
		test_losses = []
		first_layer_test_losses = []	
		for batch_id, sentences in enumerate(tqdm(test_dl, total=len(test_dl), disable = True)):
			curr_sent = sentences.to(torch.device(device))
			predcell.init_vars(batch_size = sentences.shape[0])
			loss, first_layer_loss, predictions = predcell.forward(curr_sent)
			test_losses.append(loss.detach())
			first_layer_test_losses.append(torch.mean(first_layer_loss.detach()).numpy())
		mean_test_losses = np.mean(test_losses)
		mean_first_layer_test_losses = np.mean(first_layer_test_losses)
		all_test_loss.append(mean_test_losses)

	print ("Training losses = " + str(all_train_loss))
	print ("Testing losses = " + str(all_test_loss))
	
	fig = plt.figure()
	ax = fig.gca()
	ax.plot(all_train_loss, label = "Training Loss")
	ax.plot(all_test_loss, label = "Validation Loss")
	ax.legend()
	fig.savefig("Losses.png", format = "png", dpi = 1000, transparent = True)
	plt.clf()

	return

def get_text(file_name):
	path = file_name
	with io.open(path, encoding="utf-8") as f:
		text = f.read().lower()
	text = text.replace("\n", " ")
	return text

if __name__ == "__main__":
	main()
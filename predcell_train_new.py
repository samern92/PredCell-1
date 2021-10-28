# predcell_train_new.py

import io
import sys
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Text_Loader import *
from predcell_subtractive_relu import *

np.random.seed(0)

# Corpus Settings
TEXT_FILE = "./nietzsche.txt" # consider changing corpus
TRAIN_TEST_PROP = [5,2]
DL_PARAMS = {"batch_size": 2048, "shuffle": True} #batch size is a hyperparameter
MAXLEN = len("the quick brown fox jumps over the lazy dog ") # this is a hyperparameter
STEP = 5
# Model Settings
CHECKPOINT = False
NUM_LSTMS = 2
NUM_EPOCHS = 1000
CYCLE_LENGTH = 5 # this is a hyperparameter; also, we shouldn't necessarily train layers 1 and 2 for the same duration
PERIODIC_TRAINING_ENABLED = True
if PERIODIC_TRAINING_ENABLED and NUM_LSTMS != 2:
	raise RuntimeError('Periodic training is meant for a 2 layer model.')
# Monitoring Settings
MONITOR_INTERVAL = [100,40]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
# device = torch.device("cpu")
print ("Using " + str(torch.cuda.device_count()) + " devices\n") 

def main():

# Load dataset
################################################################################

	# FNT_text = Text_Loader(get_text(TEXT_FILE)[:50], MAXLEN, STEP)
	# FNT_train, FNT_test = FNT_text.split_data(proportion = TRAIN_TEST_PROP)
	# train_dl = DataLoader(FNT_train, **DL_PARAMS)
	# test_dl = DataLoader(FNT_test, **DL_PARAMS)

	train_text,test_text,onehot_details = split_text(get_text(TEXT_FILE), proportion = TRAIN_TEST_PROP)
	FNT_train = Text_Loader(train_text, MAXLEN, STEP, onehot_details = onehot_details)
	FNT_test = Text_Loader(test_text, MAXLEN, STEP, onehot_details = onehot_details)
	train_dl = DataLoader(FNT_train, **DL_PARAMS)
	test_dl = DataLoader(FNT_test, **DL_PARAMS)

	print ("Training info:\n\tsize = " + str(len(FNT_train)) + "\n\tbatch = " + str(len(train_dl)))
	print ("Training info:\n\tsize = " + str(len(FNT_test)) + "\n\tbatch = " + str(len(test_dl)))

# Define Model
################################################################################

	predcell = PredCells(NUM_LSTMS + 1, MAXLEN, 128, FNT_train.n_chars) # 128 is a hyperparameter
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
	optimizer = torch.optim.Adam(trainable_params, lr=8e-4) # this is a hyperparameter
	start_epoch = 0
	# load checkpoint
	CHECKPOINT = True
	checkpoint_path = './checkpoints/checkpoint_epoch_999.pt'
	checkpoint = torch.load(checkpoint_path)
	predcell.load_model(checkpoint_path)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	start_epoch = checkpoint['epoch']
	
	train_counter = 0
	test_counter = 0
	# orig_monitor_train_loss_1 = monitor_train_loss_1 if CHECKPOINT else []
	# orig_monitor_train_loss_2 = monitor_train_loss_2 if CHECKPOINT else []
	# orig_monitor_test_loss_1 = monitor_test_loss_1 if CHECKPOINT else []
	# orig_monitor_test_loss_2 = monitor_test_loss_2 if CHECKPOINT else []
	monitor_train_loss_1 = []
	monitor_test_loss_1 = []
	monitor_train_loss_2 = []
	monitor_test_loss_2 = []
	monitor_pred = []
	true_pred = []
	batch_id_records = []


	all_train_loss = []
	all_test_loss = []
	for epoch in trange(NUM_EPOCHS):
	# for epoch in range(NUM_EPOCHS):
		# print ("\nEpoch: " + str(epoch) + "\n################################################################################\n")
		if PERIODIC_TRAINING_ENABLED:
			cycle = epoch % (2 * CYCLE_LENGTH)
			if cycle == 0:
				# Train 2, freeze 3
				# print('>> Training layer 2, freezing layer 3 <<')

				# Enable/disable the losses
				predcell.layer_losses_enabled[0] = True
				predcell.layer_losses_enabled[1] = False

				# Enable/disable training
				predcell.enable_layer_training(1)
				predcell.disable_layer_training(2)
			elif cycle == CYCLE_LENGTH:
				# Train 3, freeze 2
				# print('>> Training layer 3, freezing layer 2 <<')

				# Enable/disable the losses
				predcell.layer_losses_enabled[0] = False
				predcell.layer_losses_enabled[1] = True

				# Enable/disable training
				predcell.disable_layer_training(1)
				predcell.enable_layer_training(2)

		# train
		# print ("Start Training")
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
			first_layer_train_losses.append(torch.mean(first_layer_loss.cpu().detach()).numpy())
			train_counter += 1
			if train_counter % MONITOR_INTERVAL[0] == 0:
				batch_id_records.append(train_counter)
				if cycle == 0:
					monitor_train_loss_1.append(train_losses[-1])
				elif cycle == CYCLE_LENGTH:
					monitor_train_loss_2.append(train_losses[-1])

				# curr_predict = []
				# true_pred.append(sentences[0])
				# for time_points in predictions:
				# 	curr_predict.append(time_points[0].detach().numpy())
				# curr_predict = np.array(curr_predict)
				# monitor_pred.append(curr_predict)


				monitor_pred.append(predictions[0])
				# print (predictions[0].shape)
				# print (np.argmax(predictions[0], axis = 1))
				true_pred.append(sentences[0])
		# mean_train_losses = np.mean(train_losses.numpy())
		# mean_first_layer_train_losses = np.mean(first_layer_train_losses)
		# all_train_loss.append(mean_train_losses)

		# validation
		# print ("Start Validating")
		test_losses = []
		first_layer_test_losses = []	
		for batch_id, sentences in enumerate(tqdm(test_dl, total=len(test_dl), disable = True)):
			curr_sent = sentences.to(torch.device(device))
			predcell.init_vars(batch_size = sentences.shape[0])
			loss, first_layer_loss, predictions = predcell.forward(curr_sent)
			test_losses.append(loss.detach())
			first_layer_test_losses.append(torch.mean(first_layer_loss.cpu().detach()).numpy())
			test_counter += 1
			if test_counter % MONITOR_INTERVAL[1] == 0:
				if cycle == 0:
					monitor_test_loss_1.append(test_losses[-1])
				elif cycle == CYCLE_LENGTH:
					monitor_test_loss_2.append(test_losses[-1])

		# mean_test_losses = np.mean(test_losses)
		# mean_first_layer_test_losses = np.mean(first_layer_test_losses)
		# all_test_loss.append(mean_test_losses)

	# print ("Training losses = " + str(all_train_loss))
	# print ("Testing losses = " + str(all_test_loss))
	
	enforce_length = min([len(monitor_train_loss_1), len(monitor_train_loss_2),len(monitor_test_loss_1), len(monitor_test_loss_2)])
	monitor_train_loss_1 = monitor_train_loss_1[:enforce_length]
	monitor_train_loss_2 = monitor_train_loss_2[:enforce_length]
	monitor_test_loss_1 = monitor_test_loss_1[:enforce_length]
	monitor_test_loss_2 = monitor_test_loss_2[:enforce_length]
	batch_id_records = batch_id_records[:enforce_length]
	checkpoint_path = f'./checkpoints/checkpoint_epoch_{start_epoch + epoch}.pt'
	checkpoint = {
		'epoch': start_epoch + epoch,
		'optimizer_state_dict': optimizer.state_dict(),
		'monitor_train_loss_1': monitor_train_loss_1,
		'monitor_train_loss_2': monitor_train_loss_2,
		'monitor_test_loss_1': monitor_test_loss_1,
		'monitor_test_loss_2': monitor_test_loss_2,
		'batch_id_records': batch_id_records
	}
	predcell.save_model(checkpoint, checkpoint_path)
	fig = plt.figure()
	ax = fig.gca()
	ax.plot(monitor_train_loss_1, label = "Training Loss 1")
	ax.plot(monitor_train_loss_2, label = "Training Loss 2")
	ax.plot(monitor_test_loss_1, label = "Validation Loss 1")
	ax.plot(monitor_test_loss_2, label = "Validation Loss 2")
	# ax.set_xticks(np.arange(enforce_length))
	# ax.set_xticklabels(batch_id_records)
	ax.legend()
	fig.savefig(f"Losses_epoch_{start_epoch + epoch}.png", format = "png", dpi = 1000, transparent = True)
	plt.clf()

	with open (f"pred_results_{start_epoch + epoch}.txt", "w") as outfile:
		for ind in range(len(batch_id_records)):
			outfile.write("Batch: " + str(batch_id_records[ind]) + "\n")
			outfile.write("Pred:  " + "".join(onehot_to_char(monitor_pred[ind], FNT_train.ind_to_char_dict)) + "\n")
			outfile.write("True: " + "".join(onehot_to_char(true_pred[ind], FNT_train.ind_to_char_dict)) + "\n\n")
	return

def get_text(file_name):
	path = file_name
	with io.open(path, encoding="utf-8") as f:
		text = f.read().lower()
	text = text.replace("\n", " ")
	return text



if __name__ == "__main__":
	main()

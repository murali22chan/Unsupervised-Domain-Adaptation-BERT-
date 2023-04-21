from transformers import BertModel, BertTokenizer
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import os
import json
import pandas as pd
import wandb
import torch.nn as nn
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Parameters
epochs = 3
batch_size = 6
device = 'cuda'


class BertClassifier(nn.Module):
	def __init__(self, num_labels):
		super(BertClassifier, self).__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

	def forward(self, input_ids, attention_mask=None, token_type_ids=None):
		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids)
		pooled_output = outputs[1]
		logits = self.classifier(pooled_output)
		return logits

class ReverseLayerF(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha
		return output, None

class BertDomainAdaptation(nn.Module):
	def __init__(self, num_labels, alpha, beta):
		super(BertDomainAdaptation, self).__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
		self.human_machine_clf = nn.Linear(self.bert.config.hidden_size, 2)
		self.alpha = alpha
		self.beta = beta

	def forward(self, source_input_ids, source_attention_mask=None, target_input_ids=None, target_attention_mask=None, human_machine_labels=None):

		if source_input_ids is not None:
			source_outputs = self.bert(source_input_ids, attention_mask=source_attention_mask)
			source_pooled_output = source_outputs[1]
			reverse_src_pooled_output = ReverseLayerF.apply(source_pooled_output, self.alpha)
			src_logits = self.classifier(reverse_src_pooled_output)
			human_machine_logits = self.human_machine_clf(source_pooled_output)
			human_machine_loss_fn = nn.CrossEntropyLoss()
			human_machine_loss = human_machine_loss_fn(human_machine_logits, human_machine_labels)


		if target_input_ids is not None:
			target_outputs = self.bert(target_input_ids, attention_mask=target_attention_mask)
			target_pooled_output = target_outputs[1]
			reverse_tgt_pooled_output = ReverseLayerF.apply(target_pooled_output, self.alpha)
			tgt_logits = self.human_machine_clf(reverse_tgt_pooled_output)
			tgt_labels = reverse_tgt_pooled_output.new_ones(reverse_tgt_pooled_output.shape[0]).detach()

			src_labels = reverse_src_pooled_output.new_zeros(reverse_src_pooled_output.shape[0]).detach()

			src_labels = src_labels.type(torch.cuda.LongTensor)
			tgt_labels = tgt_labels.type(torch.cuda.LongTensor)

			src_labels.to(device)
			tgt_labels.to(device)
			src_logits.to(device)
			tgt_logits.to(device)

			loss_fn = torch.nn.CrossEntropyLoss()
			loss = loss_fn(src_logits, src_labels)
			loss +=loss_fn(tgt_logits, tgt_labels)

			return human_machine_loss + self.beta * loss , human_machine_logits

		return human_machine_loss, human_machine_logits

#Function to split the training data as training, validation and testing
def split_data_training_validation_testing(train_df, test_df, split_ratio = 0.4):
	train_text, val_text, train_labels, val_labels = train_test_split(train_df['answer'], train_df['label'], random_state=0, test_size= split_ratio, stratify=train_df['label'])
	test_text = test_df['answer']
	test_labels = test_df['label']

	return train_text, train_labels, val_text, val_labels, test_text, test_labels


#Function to split the training data as training, validation and testing for same domains
def split_data_training_validation_testing_same_domain(train_df):
	train_text, temp_text, train_labels, temp_labels = train_test_split(train_df['answer'], train_df['label'], 
																	random_state=0, 
																	test_size=0.4,
																	stratify= train_df['label'])
	# TODO: Make Sure the same domain and cross-domain should utilize the same amount of test dataset
	val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
																random_state=0, 
																test_size=0.5, 
																stratify=temp_labels)
	return train_text, train_labels, val_text, val_labels, test_text, test_labels

#Function to create a dataset loader
def create_dataset_loader(train_text, train_labels, val_text, val_labels, test_text, test_labels, train_sequence_length, test_sequence_length, batch_size = 6):

	tokens_train = tokenizer.batch_encode_plus(
		train_text.tolist(),
		max_length = train_sequence_length,
		pad_to_max_length=True,
		truncation=True
		)

	tokens_val = tokenizer.batch_encode_plus(
		val_text.tolist(),
		max_length = train_sequence_length,
		pad_to_max_length=True,
		truncation=True
		)

	tokens_test = tokenizer.batch_encode_plus(
		test_text.tolist(),
		max_length = test_sequence_length,
		pad_to_max_length=True,
		truncation=True
		)

	train_seq = torch.tensor(tokens_train['input_ids'])
	train_mask = torch.tensor(tokens_train['attention_mask'])
	train_y = torch.tensor(train_labels.tolist())

	val_seq = torch.tensor(tokens_val['input_ids'])
	val_mask = torch.tensor(tokens_val['attention_mask'])
	val_y = torch.tensor(val_labels.tolist())

	test_seq = torch.tensor(tokens_test['input_ids'])
	test_mask = torch.tensor(tokens_test['attention_mask'])
	test_y = torch.tensor(test_labels.tolist())

	train_data = TensorDataset(train_seq, train_mask, train_y)

	# sampler for sampling the data during training
	train_sampler = RandomSampler(train_data)

	# dataLoader for train set
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

	# wrap tensors
	val_data = TensorDataset(val_seq, val_mask, val_y)

	# sampler for sampling the data during training
	val_sampler = SequentialSampler(val_data)

	# dataLoader for validation set
	val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

	# wrap tensors
	test_data = TensorDataset(test_seq, test_mask, test_y)

	# sampler for sampling the data during testing
	test_sampler = SequentialSampler(test_data)

	# dataLoader for test set
	test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size= 1)

	return train_dataloader, val_dataloader, test_dataloader

#Function to create a dataset loader
def create_dataset_loader_DA(train_text, train_labels, val_text, val_labels, train_text_cd, train_labels_cd, train_sequence_length, test_sequence_length, batch_size = 6):

	tokens_train = tokenizer.batch_encode_plus(
		train_text.tolist(),
		max_length = train_sequence_length,
		pad_to_max_length=True,
		truncation=True
		)

	tokens_val = tokenizer.batch_encode_plus(
		val_text.tolist(),
		max_length = train_sequence_length,
		pad_to_max_length=True,
		truncation=True
		)

	tokens_train_cd = tokenizer.batch_encode_plus(
		train_text_cd.tolist(),
		max_length = train_sequence_length,
		pad_to_max_length=True,
		truncation=True
		)

	train_seq = torch.tensor(tokens_train['input_ids'])
	train_mask = torch.tensor(tokens_train['attention_mask'])
	train_y = torch.tensor(train_labels.tolist())

	val_seq = torch.tensor(tokens_val['input_ids'])
	val_mask = torch.tensor(tokens_val['attention_mask'])
	val_y = torch.tensor(val_labels.tolist())

	train_cd_seq = torch.tensor(tokens_train_cd['input_ids'])
	train_cd_mask = torch.tensor(tokens_train_cd['attention_mask'])
	train_cd_y = torch.tensor(train_labels_cd.tolist())

	train_data = TensorDataset(train_seq, train_mask, train_y)

	# sampler for sampling the data during training
	train_sampler = RandomSampler(train_data)

	# dataLoader for train set
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


	train_data_cd = TensorDataset(train_cd_seq, train_cd_mask)

	# sampler for sampling the data during training
	train_sampler_cd = RandomSampler(train_data_cd)

	# dataLoader for train set
	train_dataloader_cd = DataLoader(train_data_cd, sampler=train_sampler_cd, batch_size=batch_size)

	# wrap tensors
	val_data = TensorDataset(val_seq, val_mask, val_y)

	# sampler for sampling the data during training
	val_sampler = SequentialSampler(val_data)

	# dataLoader for validation set
	val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


	return train_dataloader, val_dataloader, train_dataloader_cd

#Function to evaluate the model
def evaluate(val_dataloader):    
	# deactivate dropout layers
	model.eval()
	total_loss, total_accuracy = 0, 0
	
	# empty list to save the model predictions
	
	total_preds = []
	
	# iterate over batches
	
	for step,batch in enumerate(val_dataloader):
			
		# push the batch to gpu
		batch = [t.to(device) for t in batch]
		sent_id, mask, labels = batch
		
		# deactivate autograd
		with torch.no_grad():
			# model predictions
			loss, preds = model(source_input_ids = sent_id, source_attention_mask=mask, target_input_ids=None, target_attention_mask=None, human_machine_labels=labels)
		
			total_loss = total_loss + loss.item()

			preds = F.softmax(preds, dim=1)
		
			preds = preds.detach().cpu().numpy()
		
			total_preds.append(preds)
		
		# compute the validation loss of the epoch
			
	avg_loss = total_loss / len(val_dataloader) 
	
	# reshape the predictions in form of (number of samples, no. of classes)
		
	total_preds  = np.concatenate(total_preds, axis=0)

	return avg_loss, total_preds

#Function to test the model
def test(val_dataloader):    
	# deactivate dropout layers
	model.eval()
	total_loss, total_accuracy = 0, 0
	
	# empty list to save the model predictions
	
	total_preds = []
	
	# iterate over batches
	
	for step,batch in enumerate(val_dataloader):
			
		# push the batch to gpu
		batch = [t.to(device) for t in batch]
		sent_id, mask, labels = batch
		
		# deactivate autograd
		with torch.no_grad():
			# model predictions
			loss, preds = model(source_input_ids = sent_id, source_attention_mask=mask, target_input_ids=None, target_attention_mask=None, human_machine_labels=labels)
		
			total_loss = total_loss + loss.item()

			preds = F.softmax(preds, dim=1)
		
			preds = preds.detach().cpu().numpy()
		
			total_preds.append(preds)
		
		# compute the validation loss of the epoch
			
	avg_loss = total_loss / len(val_dataloader) 
	
	# reshape the predictions in form of (number of samples, no. of classes)
		
	total_preds  = np.concatenate(total_preds, axis=0)

	return avg_loss, total_preds

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--trainDomain', type=str, default="finance")
	parser.add_argument('--testDomain', type=str, default="medicine")
	parser.add_argument('--trainSeqLength', type=int, default="128")
	parser.add_argument('--testSeqLength', type=int, default="32")
	parser.add_argument("--is_downsample", action="store_true")



	args = parser.parse_args()
	trainSequenceLength = args.trainSeqLength
	testSequenceLength = args.testSeqLength
	# log the experiment settings
	wandb.init(project="domain-adaptation-usda-trial", config=args)

	print("Experiment Details")
	print("Training Data: "+str(args.trainDomain)+" Testing Data: "+str(args.testDomain)+" Training Sequnce Length: "+str(args.trainSeqLength)+" Testing Sequence Length: "+str(args.testSeqLength))


	#Loading the domain csv
	train_df = pd.read_csv(args.trainDomain + "_full.csv")
	cross_domain_df = pd.read_csv(args.testDomain + "_full.csv")

	#If training and testing domain are same then we have to get the testing data from training data itself.
	if args.trainDomain == args.testDomain:
		sys.exit(0)
		# train_text, train_labels, val_text, val_labels, test_text, test_labels = split_data_training_validation_testing_same_domain(train_df)
	else:
		# TODO: MAKE SURE cross-domain and in-domain utilize the same test dataset. Your previous setting will utilize the whole dataset for evaluation.
		# The test dataset should be the same across cases.
		train_text, train_labels, val_text, val_labels, _, _ = split_data_training_validation_testing_same_domain(train_df)
		train_text_cd, train_labels_cd, val_text_cd, val_labels_cd, test_text, test_labels = split_data_training_validation_testing_same_domain(cross_domain_df)



	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	domain_1_samples = len(train_text)
	cross_domain_samples = len(train_text_cd)
	if args.is_downsample:
		small_value = domain_1_samples if domain_1_samples > cross_domain_samples else cross_domain_samples
		train_text = train_text[:small_value]
		train_labels = train_labels[:small_value]
		val_text = val_text[:small_value]
		val_labels = val_labels[:small_value]
		train_text_cd = train_text_cd[:small_value]
		train_labels_cd = train_labels_cd[:small_value]




	iterations = (max(domain_1_samples, cross_domain_samples) / batch_size)

	_, _, test_dataloader = create_dataset_loader(train_text, train_labels, val_text, val_labels, test_text, test_labels, trainSequenceLength, testSequenceLength, batch_size)
	train_dataloader, val_dataloader, train_dataloader_cd = create_dataset_loader_DA(train_text, train_labels, val_text, val_labels,train_text_cd, train_labels_cd, trainSequenceLength, testSequenceLength, batch_size)

	lager_data_loader = train_dataloader if domain_1_samples > cross_domain_samples else train_dataloader_cd
	small_data_loader = train_dataloader_cd if domain_1_samples > cross_domain_samples else train_dataloader
	small_data_iter = iter(small_data_loader)
	is_large_src =  domain_1_samples > cross_domain_samples

	model = BertDomainAdaptation(num_labels=2, alpha=0.5, beta = 1)
	model.cuda()
	optimizer = torch.optim.Adam(model.parameters())

	config = wandb.config
	config.learning_rate = 1e-5
	config.num_epochs = epochs
	config.batch_size = batch_size


	wandb.run.name = "bert_train_"+str(args.trainDomain)+"_test_"+str(args.testDomain)+"_trainSeqLen_"+str(trainSequenceLength)+"_testSeqLen_"+str(testSequenceLength)

	wandb.watch(model, log="all")

	best_valid_acc = 0

	train_losses=[]
	valid_losses=[]


	for epoch in tqdm.tqdm(range(epochs)):

		model.train()

		train_total_loss = 0

		print("Epoch: ",epoch+1)

		#Training the classifier
		for step,batch in tqdm.tqdm(enumerate(lager_data_loader)):

			batch = [r.to(device) for r in batch]
			try:
				small_batch = next(small_data_iter)
			except:
				small_data_iter = iter(small_data_loader)
				small_batch = next(small_data_iter)



			# sent_id, mask, labels = batch
			# clear previously calculated gradients 
			model.zero_grad()

			if is_large_src is False:
				small_batch = [r1.to(device) for r1 in small_batch]
				sent_id, mask, labels = small_batch
				sent_id_tgt, mask_tgt = batch
			else:
				small_batch = [r1.to(device) for r1 in small_batch]
				sent_id, mask, labels = batch
				sent_id_tgt, mask_tgt = small_batch

			loss, preds_logits = model(source_input_ids = sent_id, source_attention_mask = mask,target_input_ids = sent_id_tgt, target_attention_mask = mask_tgt, human_machine_labels = labels)

			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			train_total_loss+=loss.item()
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			

		#Calculating valdiation accuracy
		valid_loss, val_preds = evaluate(val_dataloader)
		preds_val = np.argmax(val_preds, axis = 1)
		val_y = torch.tensor(val_labels.tolist())
		val_report = classification_report(val_y, preds_val, output_dict=True)
		valid_accuracy = val_report['accuracy']


		#save the best model
		if valid_accuracy > best_valid_acc:
			best_valid_acc = valid_accuracy
			model_name = "bert_train_"+str(args.trainDomain)+"_test_"+str(args.testDomain)+"_trainSeqLen_"+str(trainSequenceLength)+"_testSeqLen_"+str(testSequenceLength)+".pt"
			checkpoint = {'state_dict': model.state_dict(), 'best_val_acc': best_valid_acc, 'optimizer': optimizer.state_dict() }
			torch.save(checkpoint, "models/"+model_name)
		# append training and validation loss
		train_losses.append(train_total_loss)
		valid_losses.append(valid_loss)
		wandb.log({"epoch":epoch+1,"training_loss": train_total_loss, "validation_loss": valid_loss, "validation_accuray":valid_accuracy})




	average_loss, total_preds = test(test_dataloader) #Testing

	#Getting the metrics
	preds = np.argmax(total_preds, axis = 1)
	test_y = torch.tensor(test_labels.tolist())
	report = classification_report(test_y, preds, output_dict=True)

	macro_precision =  report['macro avg']['precision'] 
	macro_recall = report['macro avg']['recall']    
	macro_f1 = report['macro avg']['f1-score']

	accuracy = report['accuracy']

	#Logging to Wandb
	wandb.log({"accuracy": accuracy, "precision": macro_precision, "recall": macro_recall,"f1-score": macro_f1})

	#Saving experiments result to local directory as json
	results = {"training_data": args.trainDomain,"testing_data": args.testDomain,"training_seq_len": args.trainSeqLength,"testing_seq_len": args.testSeqLength,"accuracy": accuracy, "precision": macro_precision, "recall": macro_recall,"f1-score": macro_f1}

	json_data = json.dumps(results)
	directory = "results/"
	with open(os.path.join(directory, model_name+".json"), "w") as f:
		f.write(json_data)

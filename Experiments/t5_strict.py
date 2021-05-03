# -*- coding: utf-8 -*-
"""T5 baseline on strict"""
# =============== Import Modules ==============
import random
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging as log
import time

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import (EvalPrediction, AdamW, get_linear_schedule_with_warmup, 
                          T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer)
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# =============== Self Defined ===============
from args import args, check_args


def main():
    start = time.time()
    parser = args.parse_args()

	# run some checks on arguments
    check_args(parser)
    log_name = os.path.join(parser.run_log, '{}_run_log_{}.log'.format(parser.experiment,dt.now().strftime("%Y%m%d_%H%M")))
    log.basicConfig(filename=log_name, format='%(asctime)s | %(name)s -- %(message)s', level=log.INFO)
    os.chmod(log_name, parser.access_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting experiment {} T5 baseline on {}.".format(parser.experiment,device))
    log.info("Starting experiment {} T5 baseline on {}.".format(parser.experiment,device))

	# get saved models dir
    base_saved_models_dir = parser.save_dir
    saved_models_dir = os.path.join(base_saved_models_dir)
    log.info("We will save the models in this directory: {}".format(saved_models_dir))

	# get data dir
    main_data_path = parser.data_dir

    df = pd.read_csv(main_data_path+'/circa-data_strict.tsv', sep='\t', index_col='id')

    class DataSetClass(Dataset):

        def __init__(self, dataframe, tokenizer, qa_pair_len, target_len, qa_pair_text, target_text):
            self.tokenizer = tokenizer
            self.data = dataframe
            self.qa_pair_len = qa_pair_len
            self.summ_len = target_len
            self.target_text = self.data[target_text]
            self.qa_pair_text = self.data[qa_pair_text]

        def __len__(self):
            return len(self.target_text)

        def __getitem__(self, index):
            qa_pair_text = str(self.qa_pair_text[index])
            target_text = str(self.target_text[index])

            #cleaning data so as to ensure data is in string type
            qa_pair_text = ' '.join(qa_pair_text.split())
            target_text = ' '.join(target_text.split())

            source = self.tokenizer.batch_encode_plus([qa_pair_text], max_length= self.qa_pair_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
            target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

            source_ids = source['input_ids'].squeeze()
            source_mask = source['attention_mask'].squeeze()
            target_ids = target['input_ids'].squeeze()
            target_mask = target['attention_mask'].squeeze()

            return {
                'source_ids': source_ids.to(dtype=torch.long), 
                'source_mask': source_mask.to(dtype=torch.long), 
                'target_ids': target_ids.to(dtype=torch.long),
                'target_ids_y': target_ids.to(dtype=torch.long)
            }


    def train(epoch, tokenizer, model, device, loader, optimizer):

        model.train()
        for _,data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def validate(epoch, tokenizer, model, device, loader):

        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, data in enumerate(loader, 0):
                y = data['target_ids'].to(device, dtype = torch.long)
                ids = data['source_ids'].to(device, dtype = torch.long)
                mask = data['source_mask'].to(device, dtype = torch.long)

                generated_ids = model.generate(
                    input_ids = ids,
                    attention_mask = mask, 
                    max_length=150, 
                    num_beams=2,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
                    )
                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
                if _%10==0:
                    print(f'Completed {_}')

                predictions.extend(preds)
                actuals.extend(target)
        return predictions, actuals

    def T5Trainer(dataframe, qa_pair_text, target_text, model_params, output_dir="./t5_outputs/" ):

        torch.manual_seed(model_params["SEED"])
        np.random.seed(model_params["SEED"])
        torch.backends.cudnn.deterministic = True

        tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

        model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
        model = model.to(device)

        dataframe = dataframe[[qa_pair_text,target_text]]


        # 60% of the data will be used for training and the rest for test and dev. 
        train_size = 0.6
        train_dataset=dataframe.sample(frac=train_size,random_state = model_params["SEED"])
        val_dataset=dataframe.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)

        val_size = 0.5
        dev_dataset=val_dataset.sample(frac=val_size,random_state = model_params["SEED"])
        test_dataset=val_dataset.drop(dev_dataset.index).reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        training_set = DataSetClass(train_dataset, tokenizer, model_params["MAX_qa_pair_text_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], qa_pair_text, target_text)
        dev_set = DataSetClass(dev_dataset, tokenizer, model_params["MAX_qa_pair_text_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], qa_pair_text, target_text)
        test_set = DataSetClass(test_dataset, tokenizer, model_params["MAX_qa_pair_text_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], qa_pair_text, target_text)


        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': model_params["TRAIN_BATCH_SIZE"],
            'shuffle': True,
            'num_workers': 0
            }


        dev_params = {
            'batch_size': model_params["DEV_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': 0
            }

        test_params = {
            'batch_size': model_params["TEST_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': 0
            }

        training_loader = DataLoader(training_set, **train_params)
        dev_loader = DataLoader(dev_set, **dev_params)
        test_loader = DataLoader(test_set, **test_params)

        optimizer = torch.optim.Adam(params = model.parameters(), lr=model_params["LEARNING_RATE"])

        # Training
        for epoch in range(model_params["TRAIN_EPOCHS"]):
            train(epoch, tokenizer, model, device, training_loader, optimizer)
        path = os.path.join(output_dir, "model_files")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)


        # Evaluating 
        console.log(f"[Initiating Dev]...\n")
        for epoch in range(model_params["DEV_EPOCHS"]):
            predictions, actuals = validate(epoch, tokenizer, model, device, dev_loader)
            final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
            final_df.to_csv(os.path.join(output_dir,'predictions.csv'))

    model_params={
        "MODEL":"t5-base",             # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE":32,          # training batch size
        "DEV_BATCH_SIZE":32,            # dev batch size
        "TEST_BATCH_SIZE":32,            # test batch size
        "TRAIN_EPOCHS":3,              # number of training epochs
        "DEV_EPOCHS":1,                # number of dev epochs
        "TEST_EPOCHS":1,                # number of test epochs
        "LEARNING_RATE":1e-4,          # learning rate
        "MAX_qa_pair_text_LENGTH":512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH":50,   # max length of target text
        "SEED": 42                     # set seed for reproducibility 

    }

    T5Trainer(dataframe=df, qa_pair_text="YN_s", target_text="goldstandard1", model_params=model_params)



    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(preds, labels):
        label_dict_inverse = {v: k for k, v in strict_dict.items()}
        
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__ == "__main__":
    main()
#!/usr/bin/env python

import os
import shutil
import json
import h5py
import keras
import MeCab
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

def load_datasets(x_input_file_path, y_input_file_path):
  input_x = open(x_input_file_path).readlines()
  input_y = open(y_input_file_path).readlines()

  tagger = MeCab.Tagger("-Owakati")

  train_x = []
  train_y = [int(line_y) for line_y in input_y]

  for line_x in input_x:
    train_x.append(f'<s>{tagger.parse(line_x).strip()}</s>')

  tokenizer_x = Tokenizer(filters="")
  tokenizer_x.fit_on_texts(train_x)
  train_x = tokenizer_x.texts_to_sequences(train_x)
  train_x = pad_sequences(train_x, padding='post')

  train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)

  return train_x, train_y, test_x, test_y, tokenizer_x


def train(train_x, train_y, test_x, test_y, tokenizer_x, output_dir_path):
  label_num = len(pd.Series(train_y).unique())
  vocab_size_x = len(tokenizer_x.word_index) + 1

  train_y = to_categorical(train_y, label_num)
  test_y = to_categorical(test_y, label_num)

  tbcb = TensorBoard(log_dir=output_dir_path, histogram_freq=0, write_graph=True)

  model = Sequential()
  model.add(Embedding(vocab_size_x, 256, mask_zero=True))
  model.add(LSTM(256))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(label_num, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  result = model.fit(
    x=train_x,
    y=train_y,
    validation_data=(test_x, test_y),
    epochs=10,
    batch_size=50,
    callbacks=[tbcb]
  )

  scores = model.evaluate(test_x, test_y, verbose=0)
  print(f'Accuracy: {scores[1]*100}')

  if os.path.isdir(output_dir_path):
    shutil.rmtree(output_dir_path)
  os.mkdir(output_dir_path)

  with open(f'{output_dir_path}/model.json', 'w') as model_file:
    model_file.write(model.to_json())
  with open(f'{output_dir_path}/dictionary.json', 'w') as dictionary_file:
    json.dump(tokenizer_x.word_index, dictionary_file)
  with open(f'{output_dir_path}/train.history', 'wb') as history_file:
    pickle.dump(result.history, history_file)
  model.save_weights(f'{output_dir_path}/param.hdf5')
  plot_model(model, to_file=f'{output_dir_path}/model.png')

  plt.plot(result.history['loss'], "o-", label="loss",)
  plt.plot(result.history['val_loss'], "o-", label="val_loss")
  plt.plot(result.history['acc'], "o-", label="acc")
  plt.plot(result.history['val_acc'], "o-", label="val_acc")
  plt.title('train history')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(loc="center right")
  plt.savefig(f'{output_dir_path}/plt.png')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-x', required=True)
  parser.add_argument('--input-y', required=True)
  parser.add_argument('--output', default='./result')

  args = parser.parse_args()

  train_x, train_y, test_x, test_y, tokenizer_x = load_datasets(args.input_x, args.input_y)
  train(train_x, train_y, test_x, test_y, tokenizer_x, args.output)


if __name__ == '__main__':
  main()

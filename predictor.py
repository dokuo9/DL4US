#!/usr/bin/env python

import json
import argparse
import keras
import MeCab
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

def text_to_sequence_with_dic(text, dictionary):
  tagger = MeCab.Tagger("-Owakati")
  text = tagger.parse(text).strip()
  words = kpt.text_to_word_sequence(text)

  wordIndices = []
  for word in words:
    if word in dictionary:
      wordIndices.append(dictionary[word])
    else:
      print(f'{word} not in training corpus; ignoring.')
  return wordIndices

def predict(input_text, model_path, model_params_path, dictionary_path, label_path):
  with open(dictionary_path, 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

  tokenizer = Tokenizer(filters="", num_words=len(dictionary))
  input_data = text_to_sequence_with_dic(input_text, dictionary)
  input_data = pad_sequences([input_data], padding='post')

  model_json = open(model_path).read()
  model = model_from_json(model_json)
  model.load_weights(model_params_path)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  result = model.predict(input_data)

  with open(label_path, 'r') as label_file:
    authors = json.load(label_file)
  author = authors[str(np.argmax(result))]

  print(result)
  print(f'author: {author}, score: {result[0][np.argmax(result)] * 100}')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='./result/model.json')
  parser.add_argument('--param', default='./result/param.hdf5')
  parser.add_argument('--dic', default='./result/dictionary.json')
  parser.add_argument('--label', default='./data/formatted/label.txt')
  parser.add_argument('--text', required=True)

  args = parser.parse_args() 

  predict(args.text, args.model, args.param, args.dic, args.label)

if __name__ == '__main__':
  main()


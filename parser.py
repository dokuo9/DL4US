#!/usr/bin/env python

import sys
import os
import re
import json
import shutil
import argparse

class Parser:

  def __init__(self, input_dir_path, output_dir_path):
    self.input_dir_path = input_dir_path
    self.output_dir_path = output_dir_path
    self.input_subdir_names = os.listdir(input_dir_path)

    self.inputs = {}
    self.authors = {}

    if os.path.isdir(output_dir_path):
      shutil.rmtree(output_dir_path)
    os.mkdir(output_dir_path)

    for author_code, input_subdir_name in enumerate(self.input_subdir_names):
      input_subdir_path = input_dir_path + "/" + input_subdir_name
      self.authors[author_code] = input_subdir_name

      for input_file_name in os.listdir(input_subdir_path):
        input_file_path = input_subdir_path + "/" + input_file_name
        if input_subdir_name not in self.inputs:
          self.inputs[input_subdir_name] = {}
          self.inputs[input_subdir_name]["titles"] = []

        self.inputs[input_subdir_name]["total_texts"] = 0
        self.inputs[input_subdir_name]["code"] = author_code
        self.inputs[input_subdir_name]["titles"].append({
          "texts": [],
          "path": input_file_path,
          "title": input_file_name
        })

  def __call__(self):
    with open(f'{self.output_dir_path}/label.txt', 'w') as label_file:
      label_file.write(json.dumps(self.authors))

    min_of_texts = 10**20
    for input_subdir_name in self.input_subdir_names:
      print("[" + input_subdir_name + "]")

      input = self.inputs[input_subdir_name]

      for index, title in enumerate(input["titles"]):
        input_file_path = title["path"]

        print(" -" + input_file_path)

        input_file = open(input_file_path)
        input_file_text = input_file.read()

        # remove header
        header_divider = "-------------------------------------------------------"
        header = r'^.*{0}.*{0}\n'.format(header_divider)
        input_file_text = re.sub(header, "", input_file_text, flags=(re.DOTALL))

        # remove footer
        footer = r'\n\n\n\n.*\Z'
        input_file_text = re.sub(footer, "", input_file_text, flags=(re.DOTALL))

        # remove notes
        input_file_text = re.sub(r"［＃.*］\n", "", input_file_text)
        input_file_text = re.sub(r"［＃.*］", "", input_file_text)

        # remove rubies
        input_file_text = re.sub(r"《.*?》", "", input_file_text)
        input_file_text = re.sub(r"｜", "", input_file_text)

        # remove indentation and split
        sentence_divider = r"\n　*"
        input_file_texts = re.split(sentence_divider, input_file_text, flags=(re.DOTALL))

        # remove blank row
        input_file_texts = [sentence for sentence in input_file_texts if sentence != '']

        # store prop
        self.inputs[input_subdir_name]["titles"][index]["texts"] = input_file_texts
        self.inputs[input_subdir_name]["total_texts"] += len(input_file_texts)

        input_file.close()

      if min_of_texts > self.inputs[input_subdir_name]["total_texts"]:
        min_of_texts = self.inputs[input_subdir_name]["total_texts"]

    # output file
    x_output_file_path = f'{self.output_dir_path}/data_x'
    y_output_file_path = f'{self.output_dir_path}/data_y'

    for author in self.inputs:
      texts_count = 0
      for title in self.inputs[author]["titles"]:
        for text in title["texts"]:
          if len(text) < 40 or len(text) > 300:
            continue
          if texts_count >= min_of_texts:
            break
          with open(x_output_file_path, 'a') as file:
            file.write(f'{text}\n')
          with open(y_output_file_path, 'a') as file:
            file.write(f'{self.inputs[author]["code"]}\n')
          texts_count += 1

    return f'Write files to {self.output_dir_path}'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True)
  parser.add_argument('--output', required=True)

  args = parser.parse_args()

  parser = Parser(args.input, args.output)
  result = parser()

  print(result)


if __name__ == '__main__':
  main()

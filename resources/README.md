---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': A
          '1': B
          '2': C
          '3': D
          '4': E
          '5': F
          '6': G
          '7': H
          '8': I
          '9': J
          '10': K
          '11': L
          '12': M
          '13': 'N'
          '14': O
          '15': P
          '16': Q
          '17': R
          '18': S
          '19': T
          '20': U
          '21': V
          '22': W
          '23': X
          '24': 'Y'
          '25': Z
  splits:
  - name: train
    num_bytes: 5559518
    num_examples: 520
  download_size: 5494142
  dataset_size: 5559518
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
task_categories:
- image-classification
language:
- en
tags:
- code
size_categories:
- n<1K
---
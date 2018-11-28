import pandas as pd
import numpy as np


def _get_file_list(pandas_data_frame, train_folder, file_field):
    byte_list = []
    files = pandas_data_frame[file_field]
    files = files.values.reshape(pandas_data_frame.shape[0],1)
    files = train_folder + files
    for item in files:
        single_item = item[0].encode('utf-8')
        byte_list.append(single_item)
    return byte_list

def _get_label_list(pandas_data_frame, file_field):
    temp=[]
    parsed = pandas_data_frame.drop([file_field], axis=1)
    for row in parsed.iterrows():
        index, data = row
        temp.append(data.tolist())
    return temp;

def get_files_and_labels(pandas_data_frame, train_folder,file_field):
    files = _get_file_list(pandas_data_frame, train_folder,file_field)
    labels = _get_label_list(pandas_data_frame, file_field)
    return files,labels
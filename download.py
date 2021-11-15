import os
import time
import requests
import xmltodict
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

def split_last_word(sentence, delimiter=" "):
    """ returns the sentence without the last word """
    l = sentence.split(delimiter)
    return delimiter.join(l[:-1]), l[-1]

def get_last_modified(url):
    datalink, name = split_last_word(url, "/")
    response = requests.get(datalink)
    data = xmltodict.parse(response.content)
    dicts = data['ListBucketResult']['Contents']
    last_modified = list(filter(lambda di: di['Key'] == name, dicts))[0]['LastModified']
    return datetime.fromisoformat(last_modified[:-1])

def download_data(url, name):
    """
    Downloads data from a given url and saves it as a csv file.
    If the file already exists, it checks if the previous download time is older than the last update.
    If the file doesn't exist, it downloads the data and saves it to pickle with the new updated time.
    """
    path = './data/' + name
    # check if file exists
    if os.path.isfile(path):
        # check if file is older than last update
        if os.path.getmtime(path) < get_last_modified(url).timestamp():
            # download file
            print('Downloading new data...')
            df = pd.read_csv(url)
            # save file
            df.to_pickle(path)
            print('New data downloaded.')
            return df
        else:
            print('Data is up to date.')
            df = pd.read_pickle(path)
            return df
    else:
        # download file
        print('Downloading data...')
        df = pd.read_csv(url)
        # save file
        df.to_pickle(path)
        print('Data downloaded.')
        return df
 
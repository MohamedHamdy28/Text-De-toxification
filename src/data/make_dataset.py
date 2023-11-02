
import urllib.request
import zipfile
import os


# create directory if it doesn't exist
if not os.path.exists(r'../../data/raw'):
    os.makedirs(r'../../data/raw')

# download the dataset
url = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
urllib.request.urlretrieve(url, r'../../data/raw/filtered_paranmt.zip')

# extract the dataset
with zipfile.ZipFile(r'../../data/raw/filtered_paranmt.zip', 'r') as zip_ref:
    zip_ref.extractall(r'../../data/raw')

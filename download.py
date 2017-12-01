import glob
import os
import tarfile
import wget

def download_pascal_voc_2012():
    data_dir = 'data'
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    path = '{}/{}'.format(data_dir, url.split('/')[-1])

    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(path):
        print('File exists.')
    else:
        wget.download(url, out=path)

    if os.path.exists('data/VOCdevkit'):
        print('Already extracted.')
    else:
        print('Extracting file')
        tar = tarfile.open(path)
        tar.extractall(data_dir)

if __name__ == '__main__':
    download_pascal_voc_2012()

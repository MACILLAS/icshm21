import pandas as pd
import numpy as np
import os
from os.path import isfile, join
from os import listdir
import glob
from utils import decision
"""
Create Compatible CSV for GGloud AutoML
https://cloud.google.com/vertex-ai/docs/datasets/prepare-image?_ga=2.14316789.-399339941.1624223008#csv

i.e. 
test,gs://bucket/filename1.jpeg,daisy
training,gs://bucket/filename2.gif,dandelion
gs://bucket/filename3.png
gs://bucket/filename4.bmp,sunflowers
validation,gs://bucket/filename5.tiff,tulips
"""

IMGDIR = "./image"
LABELDIR = "./label"
DEPTHDIR = "./depth"
TRAINDIR = './train'
COMPDIR = './component'
WALLDIR = os.path.join(TRAINDIR, './wall')
BEAMDIR = os.path.join(TRAINDIR, "./beam")
COLDIR = os.path.join(TRAINDIR, "./column")
WINFRAMEDIR = os.path.join(TRAINDIR, "./windowframe")
WINPANEDIR = os.path.join(TRAINDIR, "./windowpane")
BALCDIR = os.path.join(TRAINDIR, "./balcony")
SLABDIR = os.path.join(TRAINDIR, "./slab")
IGNOREDIR = os.path.join(TRAINDIR, "./ignore")
gstorage = "gs://icshm-bucket"

def appendList (proto_csv=None, files=None, label=None, gstorage_url=None):
    '''
    This function appends
    :param proto_csv:
    :param files:
    :param gstorage_url:
    :return:
    '''
    for i in range(len(files)):
        if decision(0.2):
            if decision(0.5):
                proto_csv.append(['test', join(gstorage_url, label, files[i]), label])
            else:
                proto_csv.append(['validation', join(gstorage_url, label, files[i]), label])
        else:
            proto_csv.append(['training', join(gstorage_url, label, files[i]), label])

    return proto_csv


if __name__ == "__main__":
    # Assume images are in ./train and each class is within each sub-folder
    # Create lists of Filenames from each class
    wallFiles = [f for f in listdir(join(WALLDIR)) if isfile(join(WALLDIR, f))]
    beamFiles = [f for f in listdir(join(BEAMDIR)) if isfile(join(BEAMDIR, f))]
    # we did assume we are going to look at col and beam later
    winFrameFiles = [f for f in listdir(join(WINFRAMEDIR)) if isfile(join(WINFRAMEDIR, f))]
    winPaneFiles = [f for f in listdir(join(WINPANEDIR)) if isfile(join(WINPANEDIR, f))]
    balcFiles = [f for f in listdir(join(BALCDIR)) if isfile(join(BALCDIR, f))]
    # we are not considereing slab class
    ignoreFiles = [f for f in listdir(join(IGNOREDIR)) if isfile(join(IGNOREDIR, f))]

    # Append list then convert to CSV
    proto_csv = []

    proto_csv = appendList(proto_csv, wallFiles, 'wall', gstorage)
    proto_csv = appendList(proto_csv, beamFiles, 'beam', gstorage)
    proto_csv = appendList(proto_csv, winFrameFiles, 'windowframe', gstorage)
    proto_csv = appendList(proto_csv, winPaneFiles, 'windowpane', gstorage)
    proto_csv = appendList(proto_csv, balcFiles, 'balcony', gstorage)
    proto_csv = appendList(proto_csv, ignoreFiles, 'ignore', gstorage)

    df = pd.DataFrame(proto_csv)
    df = df.sample(frac=1)

    print(df.head(100))

    df.to_csv('gsuite_csv.csv', header=False, index=False)
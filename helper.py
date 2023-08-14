import json
import argparse
import os
import sys
import time
import random
import functools
import numpy as np
import threading
import matplotlib.pyplot as plt
import math


from sklearn.feature_extraction.text import re

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

def clean_data(review):
    
    no_punc = re.sub(r'[^\w\s]', '', review)
    no_digits = ''.join([i for i in no_punc if not i.isdigit()])
    
    return(no_digits)


def ParseArgs() :
    parser = argparse.ArgumentParser(
        prog='Text Sentiment Analyser',
        description='A very basic tool that let\'s you train / use model to classify a text as positive or negative.',
        epilog='Configuration is defined in config.conf,    TSA by Ryan Ducret')

    parser.add_argument('-m', '--model', help="Specifies the path to the model to use.",
                        type=str)
    parser.add_argument('-s', '--save', help="Specifies the path to where the model will be saved.",
                        type=str)
    parser.add_argument('-t', '--train', help="Specifies the path to the training data. (enables training mode)",
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    
    args = parser.parse_args()
    global verbose
    verbose = args.verbose
    return args
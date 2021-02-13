# -*- coding: utf-8 -*-

#import part
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#verilerin alınması
veriler = pd.read_csv("veriler.csv")

print(veriler)

#veri ön işleme
boy=veriler[['boy']]

print(boy)
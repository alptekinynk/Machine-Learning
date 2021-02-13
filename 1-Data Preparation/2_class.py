# -*- coding: utf-8 -*-

#import part
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#class definition
class insan:
    boy=1.75
    kilo=80
    def VK_endeks(self,boy,kilo):
        return kilo/(boy*boy)
    
alp = insan()
print("boy: ",alp.boy,"kilo: ",alp.kilo)
print("Endeks: ",alp.VK_endeks(1.75, 80))
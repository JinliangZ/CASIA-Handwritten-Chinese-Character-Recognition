# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:19:07 2019

@author: Administrator
"""

import logging

def get_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    #create a handler to write in logger
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    #create console handder to write in console
    ch = logging.StreamHandler()
    
    #create formmater of handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    #add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    #add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger
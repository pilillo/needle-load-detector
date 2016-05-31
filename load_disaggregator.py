# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:48:39 2016

@author: andreamonacchi
"""
import numpy as np
from os import listdir
import pandas as pd
import datetime
from pytz import timezone
from scipy import signal
import matplotlib.pyplot as plt

class DatasetProcessor:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        
    def get_aggregated_signal(self):
        return self.df.sum(axis=1)
        
    def get_disaggregated_signal(self, node):
        return self.df[node]
        
    def get_node_names(self, ):
        return self.df.columns
        
    def get_dataframe(self, building, days_no):
        path = self.dataset_folder+"building"+str(building)+"/"
        self.df = pd.DataFrame()
        for d in self.select_days_from_building(building, days_no):
            self.df = self.df.append(self.load_day( path+d ))

        #self.df = self.df[self.df.index != 'timestamp'] # skip faulty lines
        self.df = self.df.sort_index()                  # sort by timestamp
        self.df.index = pd.to_datetime(self.df.index, unit="s")
        return self.df        
        
    def select_days_from_building(self, building, days):
        path = self.dataset_folder+"building"+str(building)    
        dds = listdir(path)
        if days < len(dds):
            dds = dds[:days]
        return dds
        
    def load_day(self, path):
        return pd.DataFrame.from_csv(path, header=0)
        
    def remove_holes(self):
        # Remove all NaN from the timeseries
        while self.df.isnull().any().any():
            self.df = self.df.fillna(method='bfill', limit=1)	# fill missing values with next known ones
            self.df = self.df.fillna(method='pad', limit=1)	# fill missing values with last known ones
        return self.df
        
    def remove_noise(self, sizes=[]):        
        for i, c in enumerate(self.df.columns):
            if sizes[i] > 0: self.df[c] = pd.rolling_median(self.df[c], window=sizes[i], how='median')
        #self.df = pd.rolling_median(self.ds, window=3, how='median') #signal.medfilt(self.df, 21)        
        return self.df
        
    def interpolate(self):
        #print self.df.count()
        #print self.df
        self.df = self.df.interpolate(method='time')	# interpolate timeserie
        #print self.df.count()
        #print self.df        
        return self.df
        
    def downsample_asbins(self, new_period):
        # series.resample('3T', how='sum')
        return self.df.resample(new_period, how='sum')
        
    def downsample(self, new_period):
        # http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.resample.html
        self.df = self.df.resample(new_period, how='mean')
        return self.df
        
    def uniform_period(self, sampling_frequency='1S'): # default is 1Hz
        self.df = self.df.resample(sampling_frequency, fill_method='bfill')
        return self.df
    
    def select_datetime_interval(self, start_date, end_date):
        mask = (self.df.index > start_date) & (self.df.index <= end_date)
        return self.df.loc[mask]
    
    def select_time_interval(self, start_time, end_time):
        return self.df.between_time(start_time, end_time)
    
    def get_datastore(self, path):
        self.store = pd.HDFStore(path)
    
    def add_signature(self, key, signature):
        signature.index = range(len(signature.index))   # remove the date from the signature
        self.store[key] = signature
        self.store.flush()
        
    def get_signature(self, key):
        return self.store[key]
    
    def close_datastore(self):
        self.store.close()


class BatchLoadDisaggregator:
    
    def __init__(self, templates):
        self.templates = templates
        self.max_corr = 0
        
    def disaggregate_device(self, aggregate_signal, index, beginning, min_corr):
        # look for the template t in the window
        template = self.templates[index]        
        window_size = len(template)
        window = aggregate_signal.iloc[beginning : beginning+window_size] # take a slice of the dataframe     
        return self.disaggregate_window(window, template, min_corr)
    
    def disaggregate_window(self, window, signature, min_corr):
        w = window.ix[:]
        s = signature.ix[:]
        
        window_size = len(signature)
    
        try:
            corr = np.corrcoef( w, s )[0][1]
            if abs(corr) > self.max_corr: self.max_corr = abs(corr)        
        
            if abs(corr) > min_corr:   
                print( corr )            
                plt.figure()            
                plt.plot( range(window_size), w)
                plt.plot( range(window_size), s)
                return True
            else:
                return False
        except:
            pass
        
    
    def disaggregate_dataframe(self, aggregate_signal, device_id=-1, min_corr=0.85, step_size=10):
        # todo: skip first values as big as the smallest window available
        # todo: skip last values as big as smallest window available
        self.counts = [0]*len(self.templates)
        for beginning in range(0, len(aggregate_signal.index), step_size): #range(1):
            # check if we care of only a specific device or we wanna disaggregate all
            if device_id >= 0:
                if self.disaggregate_device(aggregate_signal, device_id, beginning, min_corr):
                    self.counts[device_id] += 1
            else:
                # we wanna check all devices
                for i, t in enumerate(self.templates):
                    if t is not None:
                        if self.disaggregate_device(aggregate_signal, i, beginning, min_corr):
                            self.counts[i] += 1
            
        return self.counts
            
    def get_max_corr(self):
        return self.max_corr
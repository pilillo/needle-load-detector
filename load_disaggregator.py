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
        
    def downsample(self, new_period): #, fill_method='bfill'):
        # http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.resample.html
        # series.resample('30S', fill_method='bfill')        
        return self.df.resample(new_period, how='mean') #fill_method, how='mean')
    
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
    

# building 1    
devices = ['000D6F00036BB04C', '000D6F00029C2BD7', '000D6F000353AC8C', '000D6F0003562E10', '000D6F0003562C48', '000D6F00029C2984', '000D6F000353AE51', '000D6F0003562C0F', '000D6F0003562BF6']

dataset_folder = "/Users/andreamonacchi/Downloads/GREEND_0-2_300615/"
template_folder = "/Users/andreamonacchi/Desktop/"
        
processor = DatasetProcessor(dataset_folder)        
df = processor.get_dataframe(1, 4)

ax = df.plot(legend=False)
patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)

# preprocess the dataset
preprocessed = processor.remove_holes()     # fill with our method (pad and bfill)
preprocessed = processor.remove_noise(sizes=[30]*9) # median filter over 30 consecutive samples
#preprocessed = processor.downsample('10S')          # downsample to 1/10 Hz 
preprocessed.plot()

# get the datastore (from disk)
processor.get_datastore(template_folder+"dev_signatures.h5")

# switch between signature extraction and load disaggregation
extract = False

if extract:    
    # Extract signatures for individual devices
    #disagg = processor.get_disaggregated_signal(devices[1])
    #disagg.plot()

    # Device 0 (fridge)
    ##signature = processor.select_time_interval('08:18:40', '08:28:40')[devices[0]]
    signature0 = processor.select_datetime_interval('2014-03-12 08:18:40', '2014-03-12 08:28:40')[devices[0]]
    processor.add_signature('devices/d0', signature0)
    #print processor.get_signature('devices/d0')

    #Device 1 (dishwasher)
    signature1 = processor.select_datetime_interval('2014-03-13 10:28:30',  '2014-03-13 11:45:00')[devices[1]]
    processor.add_signature('devices/d1', signature1)

    # Device 2 (microwave)
    ##signature = processor.select_time_interval('18:24:40', '18:26:20')[devices[2]]
    signature2 = processor.select_datetime_interval('2014-03-12 18:24:40',  '2014-03-12 18:26:20')[devices[2]]
    processor.add_signature('devices/d2', signature2)
    #print processor.get_signature('devices/d2')

    # Device 3 water kettle
    signature3 = processor.select_datetime_interval('2014-03-15 08:32:00', '2014-03-15 08:42:00')[devices[3]]    
    processor.add_signature('devices/d3', signature3)

    # Device 4
    signature4 = processor.select_datetime_interval('2014-03-13 18:32:00', '2014-03-13 19:58:30')[devices[4]]    # on the second day (that is why we use datetime)
    processor.add_signature('devices/d4', signature4)

    # Device 5 (amplifier)
    #signature = processor.select_time_interval('18:24:40', '18:26:20')[devices[5]]
    #signature5 = processor.select_datetime_interval('2014-03-12 18:24:40', '2014-03-12 18:26:20')[devices[5]]
    #processor.add_signature('devices/d5', signature5)

    # Device 6 (hair dryer + charger )
    signature6 = processor.select_datetime_interval('2014-03-15 10:03:45', '2014-03-15 10:06:20')[devices[6]]
    processor.add_signature('devices/d6', signature6)
    
    # Device 7 (food processor) --> neverused
    #signature = processor.select_datetime_interval('', '')[devices[7]]
    
    # Device 8 Bedside lamp
    signature8 = processor.select_datetime_interval('2014-03-13 22:27:45', '2014-03-13 22:37:40')[devices[8]]
    processor.add_signature('devices/d8', signature8)
    
    #plt.figure()
    #signature.plot(legend=False)

else:
    # select a template and attempt the disaggregation
    templates = [processor.get_signature('devices/d0'), processor.get_signature('devices/d1'), processor.get_signature('devices/d2'), processor.get_signature('devices/d3'), processor.get_signature('devices/d4'), None, 
                 None, #processor.get_signature('devices/d6'),
                 None, None] # processor.get_signature('devices/d8') ]
    """    
    for i, t in enumerate(templates):
        if t is not None and (not t.empty):
            print t
            plt.figure()
            t.plot(title="device_"+str(i))
    """
    #plt.figure()
    agg = processor.get_aggregated_signal()
    #agg.plot()
        
    disaggregator = BatchLoadDisaggregator( templates )
    """
    d_id = 4
    counts = disaggregator.disaggregate_dataframe(agg, device_id=d_id, min_corr=0.9)
    print "Device "+str(d_id)+" detected "+str(counts[d_id])+" times"
    print "MAX CORR: ", disaggregator.get_max_corr()
    """
    counts = disaggregator.disaggregate_dataframe(agg, device_id=-1, min_corr=0.99)
    for i, c in enumerate(counts):
        print( "Device "+str(i)+" detected "+str(c)+" times" )

plt.show()

processor.close_datastore()
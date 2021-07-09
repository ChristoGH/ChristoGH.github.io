# Store market data in MongoDB!

## Introduction
I have to confess.  I am a mongodb junkie.  In my professional and private capacity.  It is _fast_, free (that is the community edition!) and convenient.  And it works especially well on my macbook!  In Python pymongo is a workhorse library for talking to MongoDB.  It is a great set of tools.

Pushing data to MongoDB makes I do not have to stress over where my data is, indexing happens there and on my laptop(!) I can happily store 100m records without issues.

In this script I describe the method I use to take data from a csv format into MongoDB.

Import some libraries for fun:

```python
from pymongo import MongoClient
client = MongoClient()
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import numpy as np
import sys,getopt

```

Specify some useful paths, I assign them to variables for future use.  

```python
original_path=os.getcwd()
original_path="""/Users/christostrydom/Documents/github_repos/ae_mp/"""
working_path="""/Users/christostrydom/Data/"""
download_path="""/Users/christostrydom/Downloads/"""

```

Define a function for parsing command line arguments:

```python
def get_opts():
    # currency,stop_loss,profit_margin,strategy=get_opts()

    return_dict={}
  
    argv = sys.argv[1:]
  
    try:
        opts, args = getopt.getopt(argv, "c:")
        for opt, arg in opts:
            if opt in ['-c']:
                return_dict['currency'] = arg 
      
    except:
        print("Error")
  

    return return_dict
```

Sometimes a \*.csv file is empty and can break code.  This function returns __False__ for an _empty_ file.

```python
def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
```

In a command line script, the following will read command line arguments into a dictionary.

```python
# return_dict=get_opts()
```

```python
CURRENCY='USDZAR' #return_dict['currency']
```

Set _db_ to be a database, call it Currencies.  This is just naming it, nothing really has happened yet:

```python
db=client.Currencies
```

Now create a __collection__, something similar to a __table__ in the sql world.  A collection contains documents, the record equivalent from sql.  In my setup, every currency is a document.  The variable *collection* now points to the 'USDZAR' table in the *Currencies* database:

```python
collection = db[CURRENCY]
```

I create a unique index on *gmt_time_py* which is a datetime variable, and an index on it simply means there cannot be duplication on this field (not impossible to have more than one entry on the same time though). This constraint could be relaxed to be an index only.  Making it an index make for fast searches on the time field, which is exactly what we are  going to do going forward. This only need to be done once!

```python
collection.create_index([ ("gmt_time_py", 1) ], unique=True)
```

In this cell I make a list of the files in the download directory. But since the upload script adds the name of the data file to each document, we can create a list of files that is sitting in the data directory that has not yet been uploaded.  Pretty neat if a tad slow but also pretty foolproof.

```python
working_path="""/Users/christostrydom/Data/{currency}/""".format(currency=CURRENCY)
# save_path="""/Users/christostrydom/Data/"""
onlyfiles = [f for f in listdir(working_path) if isfile(join(working_path, f))]
print('collect unique file names from collection.{CURRENCY}...'.format(CURRENCY=CURRENCY))
distinct_file_names=collection.distinct("file_name")
print('Done!')
onlyfiles=list(set(onlyfiles)-set(distinct_file_names))
```

The duka download file has the following headings:
'Gmt time', 'Ask', 'Bid', 'AskVolume' and 'BidVolume'.  I do the following changes and additions:
- Change Gmt time to gmt_time_py making it a pandas datetime column.  Mongo understands this,
- 'minutes' is an integer columns numbered from 1 Jan 1970 and so is 'seconds',
- 'dow' is the day of the week, as an integer,
- 'daytime' is the second value of the day,
- 'date' is the date only value,
- 'file_name' is the data file origin,
- 'currency' keeps us true.

the next cell iterates trough all the files in onlyfiles.

```python
existing_dates=[]
df=pd.DataFrame()
counter=0
for f in onlyfiles:
    counter+=1
    if (is_non_zero_file(working_path+f))&(f!='.DS_Store'):
        print("""Doing file number {counter} and {fname}""".format(counter=counter, fname=working_path+f))        
        df=pd.read_csv(filepath_or_buffer=working_path+f)    
#         gf.columns=['Gmt time', 'Ask', 'Bid', 'AskVolume', 'BidVolume']
        df.columns=['Gmt time', 'Ask', 'Bid', 'AskVolume', 'BidVolume']
        # df['gmt_time_py']=pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        df['gmt_time_py']=pd.to_datetime(df['Gmt time'], format='%Y-%m-%d %H:%M:%S.%f')
        df['minutes']=((df['gmt_time_py']-datetime(year=1970,month=1,day=1)).dt.total_seconds()/60).astype(int)
        s=df.iloc[range(1,df.shape[0])]['minutes'].values>df.iloc[range(0,(df.shape[0]-1))]['minutes'].values
        df['new_minute']=False
        df.loc[np.where(s)[0],'new_minute']=True
        df['seconds']=((df['gmt_time_py']-datetime(year=1970,month=1,day=1)).dt.total_seconds()*1000).astype(int)
        # df['just_date'] = df['gmt_time_py'].dt.date
        df['dow']=df['gmt_time_py'].dt.dayofweek
        df['daytime'] = pd.to_datetime(df['gmt_time_py']).dt.second+60*pd.to_datetime(df['gmt_time_py']).dt.minute+60*60*pd.to_datetime(df['gmt_time_py']).dt.hour
        df.sort_values('gmt_time_py',inplace=True)
        df['date']=df['gmt_time_py'].apply(lambda x: datetime(year=x.year,month=x.month,day=x.day))
        df['file_name']=f
        df['currency']=CURRENCY
        # list_df=list(df)
        df.reset_index(inplace=True,drop=True)
        # df = df.drop('index', 1)
        collection.insert_many(df.to_dict('records'))
```

And that is it! All downloaded currency tick data is now in a collection for easy access.  It can be analyzed as is, or it can be made into ohlc candles which I will do in a follow-up script.  In the following scripts it'll become clear exactly how useful this setup is.


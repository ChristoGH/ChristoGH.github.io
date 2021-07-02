# Get data!



## Introduction

Analyzing the market requires data and the more the merrier.  But market data is expensive! In this notebook, I illustrate how to download currency data __programmatically__ from here https://www.dukascopy.com/swiss/english/marketwatch/historical/ using bare bones Python, all for free!
I have already downloaded 10 years of tick data for USDZAR and EURUSD.  It is a precious resource, and I am grateful to Dukascopy for making this available.  Following the instructions below assumes you have installed the duka application from here https://pypi.org/project/duka/. This is a also another really great resource, and one I have used extensively.  It automates the process of downloading data - imagine downloading all that data 'by hand'!

I designed this to download data in business day increments, thereby avoiding weekends. 

Let's start by importing some libraries.  It makes life in Python really easy!

```python
import os
import time
from datetime import date, timedelta
import subprocess
from os import listdir
from os.path import isfile, join
import sys, getopt
```

Set two arbitrary dates, d1 and d2.  The one is the start date and the other the end date.  Pay some attention to these of course.  You want relevant data as always.

```python
d1 = date(2021,4,20)
d2 = date(2021,6,30)
```

I you want to run this as a python script, the following allows you to provide arguments, in this case the __currency__ and a seconds timer (__sleep_seconds__) 
for relaxing requests to the duka server and prevent one from being blacklisted for 'harassment'! :)

```python
def get_opts():
    # currency,stop_loss,profit_margin,strategy=get_opts()

    return_dict={}
  
    argv = sys.argv[1:]
  
    try:
        opts, args = getopt.getopt(argv, "c:s:")
      
    except:
        print("Error")
  
    for opt, arg in opts:
        if opt in ['-c']:
            return_dict['currency'] = arg
        elif opt in ['-s']:
            return_dict['sleep_seconds'] = float(arg)           
    return return_dict
```

Here is the function to parse command line arguments into a python dictionary; I commented this call out for running this in this notebook:

```python
# return_dict=get_opts()
```

```python
# currency=return_dict['currency']
# sleep_seconds=return_dict['sleep_seconds']
```

Instead let's just set the necessary values, I use my home currency __USDZAR__ here, check the duka documentations, there is a really comprehensive list of instruments there.  Also, I set sleep_seconds to 30s, I haven't had any issue with that kind of delay, I started with 300s!

```python
currency='USDZAR'
sleep_seconds=30
```

I do some directory work here, this is not necessary on your side - i suggest you dedicate a particular directory to this data though.  I use the instrument name for ease of reference. The following extracts a list of files __onlyfiles__ already downloaded to this directory.  I dont mess around in this directory - I dont want to break the code in future!

```python
original_path=os.getcwd()
working_path="""../../../Data/{currency}/""".format(currency=currency)
onlyfiles = [f for f in listdir(working_path) if isfile(join(working_path, f))]
```

Lets define here three helper functions. __download_date_fn__ extracts the dates of the files already downloaded in our working directory and give a list of dates returns only the dates now needed to be done.  This avoids duplicating downloads but it also means you can safely interrupt the process.  Always a good thing. __directory_changer__ does what it says!  __downloader_fn__ is the work horse.  It takes a list of dates and download the _tick_ data for _currency_ in day files and save them in the working director. Neat, eh?

```python
def download_date_fn(onlyfiles,ddl):
    existing_dates=[]
    for f in onlyfiles:
#         print(f)
        # print(f)
        if not f==".DS_Store":
            slist=f.split('-')[1].split('_')
            sdate=date(year=int(slist[0]),month=int(slist[1]),day=int(slist[2])).strftime("%Y-%m-%d")
    #         print(date(year=int(slist[0]),month=int(slist[1]),day=int(slist[2])).strftime("%Y-%m-%d"))
            existing_dates.append(sdate)
    return_list=list(set(ddl)-set(existing_dates))
    return_list.sort()
    return return_list  

def directory_changer(to_dir):
    try:
        # Change the current working Directory    
        os.chdir(to_dir)
        print("Directory changed to: ", to_dir)
    except OSError:
        print("Can't change the Current Working Directory")
    return 

def downloader_fn(download_date_list):
    counter = 0
    for download_date in download_date_list:
        counter+=1
        s='duka {currency} -d {download_date}'.format(download_date=download_date,
                                                      currency=currency)
        subprocess.run(s,shell=True)
        print(s)
        if counter%5==0:
            onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
            n=download_date_fn(onlyfiles=onlyfiles,ddl=download_date_list)
            print('Taking a break for {sleep_seconds} seconds.  There are {n} dates left!'.format(n=len(n),
                                                                                                  sleep_seconds=sleep_seconds))
            time.sleep(sleep_seconds) # Sleep for 3 seconds
    return
```

Create a list (ddl) of __week days__ between d1 and d2.

```python
ddl = [(d1 + timedelta(days=x)).strftime("%Y-%m-%d") for x in range((d2-d1).days + 1) if (d1 + timedelta(days=x)).isoweekday() not in [6,7] ]
```

Take ddl and derive the list of days (__download_date_list__) needed to be downloaded:

```python
download_date_list=download_date_fn(onlyfiles,ddl)
```

And here the magic happens.  Change to the working directory.  Download all the data for __download_date_list__.  Change back to our original directory.

```python
directory_changer(working_path)
downloader_fn(download_date_list)
directory_changer(original_path)
```

And that is it.  Now there are a load of csv's in your data directory ready to be explored and analyzed.  I intend to do just that in upcoming blogs.

# Create trading signals!



```python
original_path=os.getcwd()
original_path="""/Users/christostrydom/Documents/github_repos/ae_mp/"""
working_path="""/Users/christostrydom/Data/"""
download_path="""/Users/christostrydom/Downloads/"""
chart_path="""/Users/christostrydom/Documents/github_repos/ae_mp/img/"""

client = MongoClient()
```

```python
def get_opts():
    # currency,stop_loss,profit_margin,strategy=get_opts()

    return_dict={}
  
    argv = sys.argv[1:]
  
    try:
        opts, args = getopt.getopt(argv, "c:l:p:s:i:")
      
    except:
        print("Error")
  
    for opt, arg in opts:
        if opt in ['-c']:
            return_dict['currency'] = arg
        elif opt in ['-l']:
            return_dict['stop_loss'] = float(arg)
        elif opt in ['-p']:
            return_dict['profit_margin'] = float(arg)      
        elif opt in ['-s']:
            return_dict['strategy'] = arg
        elif opt in ['-i']:
            return_dict['iterations'] = float(arg)    
    return return_dict
```

Function __init_dict_fn__ looks in store_collection for the most recent tradesignal that exists for a set of strategy parameters.  This sets the starting point for the iteration below. If there are no tradesignals for a particular set of parameters, the iteration will start at the start of the database and create an entirely new set of signals.

```python
def init_dict_fn(collection, store_collection,strategy,currency,profit_margin,stop_loss):
    init_list=list(store_collection.find({'currency':currency,
                                          'profit_margin':profit_margin,
                                          'strategy':strategy,
                                          'stop_loss':stop_loss}).sort([('profit_time',-1)]).limit(1))
    if init_list:
        init_dict=init_list[0]
        search_time=init_dict['profit_time'] 
        search_price=init_dict['profit_price']
        search_id=init_dict['profit_id']
    else:
        init_dict = list(collection.find({"gmt_time_py": {"$gt": datetime(1970, 11, 12, 12)}}).sort([('gmt_time_py',1)]).limit(1))[0]
        search_time=init_dict['gmt_time_py'] 
        search_price=init_dict['Ask']
        search_id=init_dict['_id']

    return search_time,search_price,search_id
```

### Finding long trade signals:
Starting at the __start_time__, this function below looks __forward__ in time towards the first time the ask price is __lower__ than the start price MINUS a predefined stop loss value, that is, the first time a stop loss is triggered.

If a stoploss was triggered (and we know the time and price), find the first event before the time of the stoploss from above that would have triggered a profit-taking trade.  For a long trade, that would mean an bid price __above__ the start_price __plus__ some value (the profit margin).

This scenario simply means that for a start price at some start time, there was a __profit taking opportunity__ before a stop loss event.  This is then a profitable long trading opportunity and includes the cost of trading by including the bid-ask spread.

```python
def find_long_trade_fn(coll,start_time,start_price,profit_margin,stop_loss):
    profit_dict={}
    loss_dict={}
    loss_list=list(coll.find({"gmt_time_py": {"$gt": start_time,
                                             "$lt":search_time+timedelta(days=7)},
                      "Ask":{"$lt":start_price-stop_loss}}).sort([('gmt_time_py',1)]).limit(1))
    
    if (not loss_list):
        loss_list=list(coll.find({"gmt_time_py": {"$gt": start_time}}).sort([('gmt_time_py',-1)]).limit(1))
        
    if loss_list:
        loss_dict = loss_list[0]
        loss_time = loss_dict['gmt_time_py']
        profit_list = (list(coll.find({"gmt_time_py": {"$gt": start_time,
                                                       "$lt": loss_time},
                                        "Bid":{"$gt":start_price+profit_margin}
                                          }
                                         ).sort([('gmt_time_py',1)]).limit(1)))
        if profit_list:
            profit_dict=profit_list[0]
            print('found a profit trade @ {start_time} ending at {ending_time}'.format(ending_time=profit_dict['gmt_time_py'],
                                                                                      start_time=start_time))

    
    return loss_dict,profit_dict
```

### Find a profitable short trade:

This function mimics the one above for long positions:  starting at the __start_time__, this function looks __forward__ in time towards the first time the bid price is higher than the start price PLUS a predefined stop loss value, that is, the first time we will trigger a stop loss.

If a stoploss was triggered, find the first event before the time of the stoploss trade above that would have triggered a profit-taking opportunity.  For a short trade, that would mean an ask price below the start_price MINUS some value (the profit margin).

This scenario simply means that for a start price at some start time, there was a profit taking opportunity before a stop loss event.  This is then a profitable short trading opportunity, and includes the cost of trading by including the spread.

```python
def find_short_trade_fn(coll,start_time,start_price,profit_margin,stop_loss):
    profit_dict={}
    loss_dict={}
#   Starting at the start_time look forward towards the first time the bid price is higher 
#   than the start price PLUS a stop loss value, that is the first time we will trigger
#   a stoploss.
    loss_list=list(coll.find({"gmt_time_py": {"$gt": start_time,
                                             "$lt":search_time+timedelta(days=7)},
                      "Bid":{"$gt":start_price+stop_loss}}).sort([('gmt_time_py',1)]).limit(1))
    
    if (not loss_list):
        loss_list=list(coll.find({"gmt_time_py": {"$gt": start_time}}).sort([('gmt_time_py',-1)]).limit(1))
        
    if loss_list:
        loss_dict = loss_list[0]
        loss_time = loss_dict['gmt_time_py']
#       If a stoploss was triggered find the first event that would have triggered a 
# profit taking opportunity.  For a short trade that would mean and ask price below the 
# start_price MINUS some value (the profit margin)
        profit_list = (list(coll.find({"gmt_time_py": {"$gt": start_time,
                                                       "$lt": loss_time},
                                        "Ask":{"$lt":start_price-profit_margin}
                                          }
                                         ).sort([('gmt_time_py',1)]).limit(1)))
        if profit_list:
            profit_dict=profit_list[0]
            print('found a profit trade @ {start_time} ending at {ending_time}'.format(ending_time=profit_dict['gmt_time_py'],
                                                                                      start_time=start_time))

    
    return loss_dict,profit_dict
```

The script below iterates through the entire database for the currency defined by CURRENCY and notes all trading signals for long and short strategies.  It saves the signals into its own tradingsignal collection from where it can once again be extracted for a machine learning exercise.

```python
db=client.Currendcies
data_collection = db[CURRENCY]
# db.USDZAR_tradestore.drop()
signal_collection = db['{currency}_pmsl_tradestore'.format(currency=CURRENCY)]
signal_collection.create_index([ ("search_time", 1) ])

counter=0
store_dict={}
# search_time = datetime(2018, 11, 12, 12)
search_time,search_price,search_id=init_dict_fn(data_collection,
                                                signal_collection,
                                                STRATEGY,
                                                CURRENCY,
                                                PROFIT_MARGIN,
                                                STOP_LOSS)

while counter < ITERATIONS:
    counter+=1
    
    if counter%10==0:
        print(counter, search_time, search_price)#, CURRENCY,STOP_LOSS,PROFIT_MARGIN,STRATEGY)
        
    if STRATEGY=='long':
        loss_dict,profit_dict=find_long_trade_fn(data_collection,
                                                 search_time,
                                                 search_price,
                                                 PROFIT_MARGIN,
                                                 STOP_LOSS)
        if bool(profit_dict)&bool(loss_dict):
            store_dict['currency']=CURRENCY
            store_dict['strategy'] = STRATEGY        
            store_dict['profit_margin'] = PROFIT_MARGIN
            store_dict['stop_loss'] = STOP_LOSS  
            store_dict['search_time']=search_time
            store_dict['search_price']=search_price
            store_dict['search_id']=search_id
            store_dict['profit_time']=profit_dict['gmt_time_py']
            store_dict['profit_price']=profit_dict['Bid']
    #         store_dict['loss_price']=loss_price
            
            store_dict['profit_id']=profit_dict['_id']
            store_dict['_id'] = ObjectId()
    
            store_dict['date_py']=datetime(year=search_time.year, 
                                           month=search_time.month, 
                                           day=search_time.day)
    
            signal_collection.insert_one(store_dict)
    
            search_price=profit_dict['Ask']
            search_time=profit_dict['gmt_time_py']     
        if (not bool(profit_dict))&bool(loss_dict):
            search_price=loss_dict['Ask']
            search_time=loss_dict['gmt_time_py']

    if STRATEGY=='short':
        loss_dict,profit_dict=find_short_trade_fn(data_collection,
                                                 search_time,
                                                 search_price,
                                                 PROFIT_MARGIN,
                                                 STOP_LOSS)
        if bool(profit_dict)&bool(loss_dict):
            store_dict['currency']=CURRENCY
            store_dict['strategy'] = STRATEGY        
            store_dict['profit_margin'] = PROFIT_MARGIN
            store_dict['stop_loss'] = STOP_LOSS  
            store_dict['search_time']=search_time
            store_dict['search_price']=search_price
            store_dict['search_id']=search_id
            store_dict['profit_time']=profit_dict['gmt_time_py']
            store_dict['profit_price']=profit_dict['Ask']
    #         store_dict['loss_price']=loss_price
            
            store_dict['profit_id']=profit_dict['_id']
            store_dict['_id'] = ObjectId()
    
            store_dict['date_py']=datetime(year=search_time.year, 
                                           month=search_time.month, 
                                           day=search_time.day)
    
            signal_collection.insert_one(store_dict)
    
            search_price=profit_dict['Bid']
            search_time=profit_dict['gmt_time_py']     
        if (not bool(profit_dict))&bool(loss_dict):
            search_price=loss_dict['Bid']
            search_time=loss_dict['gmt_time_py']
    #     print(counter, search_time, search_price)
```

This is what the trading signals look like for 21 April 2021 for the USDZAR as an example, with a 5c profit target and a zero stop loss.  The graph below was done on tick data and only the Ask price, hence it looks somewhat unfamiliar.  But this way, we can see every single price recorded by this broker and we can simulate the real thing with some level of accuracy.

![title](img/USDZAR_5_0USDZAR_5_0_20210421.png)

And that is it!  Now there is a way of create trade signals given some expectation of profit together with a minimum stop loss to be tolerated.  Notice that our take-profit value sometimes does not cover the entire range of a sizeable move and sometimes look too be too much for attractive opportunities.  Ideally we want to cater for signals irrespective of some hard coded rule.  More on that later.

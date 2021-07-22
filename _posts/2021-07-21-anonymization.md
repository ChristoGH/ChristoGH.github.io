# Anonymization (code heavy!)

In this script, I put together the routines to anonymize a health care insurance dataset, on which I am going to perform some collaborative filtering.  Ultimately I want to land the data in a neo4j database, from where I want to do clustering and ranking of treatments, providers and members.

There are a number of columns that need to be anonymized which will otherwise breach privacy standards.  These include the company and group columns, practice number and member columns.  

I do each of these in turn.  

My goal is to make entities unique and as friendly on the human eye as possible.  For instance, member __Ms A. J. Smith__ reads *infinitely* easier than member 1097332.  And the same goes for company called __network holdings__ rather than referred to as 98440711.  My goal is to not lose any information of course, this approach requires much more work and care!  

I do anonymization in three headings.

- Company and group
- Member
- Providers

Under each of these headings a mapping file is produced.  These mapping files are the keys that close or open the information and is secured in a different location. The mapping files map anonymized columns back to the original values when joined on the anonymized datasets (and vice versa!).  

For the benefit of future privacy constraints, these files may be destroyed once a fully anonymized database has been setup.

For the benefit of anonymity the following columns are either dropped or transformed: 'member',
'dependent_no',
'first_name',
'surname',
'provider',
'doctor_name',
'company' and 'group'.
'individual'.  the goal is to the following anonymizing columns where all entries are unique.
 'fake_company_name',
 'fake_group_name',
 'fake_member',
 'fake_surname',
 'fake_name',
 'fake_provider_surname' and
 'fake_provider_name'

The treatment column is deemed to need anonymization, although even this could be treated as sensitive information.  It is left as is for now though.

I make use of two sets of data providers.  Barnum, https://pbpython.com/barnum.html can be used to create fake company names.
The names_dataset from https://github.com/philipperemy/name-dataset has a considerable set of fake names in turn, large enough not to have duplicated surnames for instance.




Import the necessary libraries:

```python
from barnum import gen_data
from random import random
import pandas as pd
import numpy as np
import os
os.chdir("""/notebooks/github_repos/fraud/""")
```

```python
from names_dataset import NameDataset
from names_dataset import NameDatasetV1
```

## Company and group anonymization

There are two features, company and group we focus on in this section.  __Group__ can be thought of as the holding company and __company__ as the subsidiary.   We clear the group and company name colums in our base dataframe of __na__ value and replace instead with __unknown_group__ and __unknown_company__ labels:

```python
df['group'].fillna(value='unknown_group', inplace=True)
df['company'].fillna(value='unknown_company', inplace=True)
```

We change the values in both __company__ and __group__ to capital letters only.  Name entry into the database was not always done in a consistent manner, hence one needs to be on the lookout for typos too!

```python
df['group'] = df['group'].str.upper()
df['company'] = df['company'].str.upper()
```

Create a unique list of all the group entities and call it __group_list__.  There are 180 unique groups (holding companies) in the database:

```python
group_list=list(set(df.group))
len(group_list)
```




    180



And likewise, create a unique list of all the company names and call it __company_list__.  There are 728 unique companies in this dataset:

```python
company_list=list(set(df.company))
len(company_list)
```




    728



Now create a list of fake company names, using the __barnum__ library, with 50% (an arbitrarily chosen number that was found to be sufficient) more than the real company names.  This is to cater for duplicated names extracted from __barnum__.  This list of unique company names contains enough entries to populate the fake company name and fake group name list:

```python
fcl=[]
for i in range(int(len(group_list)+len(company_list)*1.5)):
    fcl.append(gen_data.create_company_name())
```

Now extract a fake list of group names and companynames from the barnum list. 

```python
fake_group_list=fcl[(len(company_list)):(len(company_list)+len(group_list))]
fake_company_list=fcl[0:len(company_list)]
```

Here we verify that the fake lists contain as many entries as the real lists, without duplication:

```python
len(fake_company_list),len(set(company_list)),len(fake_group_list),len(set(group_list))
```




    (728, 728, 180, 180)



Now, create a dataframe (fake_company_df), where company names are listed with their corresponding fake names, in a 1-1 relationship:

```python
# sf=pd.DataFrame()
rows_list=[]
for company,fake_company in zip(company_list,fake_company_list):
    insert_dict={'company_name':company,
                'fake_company_name':fake_company}
    rows_list.append(insert_dict)
fake_company_df = pd.DataFrame(rows_list)   
```

And do the exact same for the group names and create a dataframe called __fake_group_df__:

```python
# sf=pd.DataFrame()
rows_list=[]
for group,fake_group in zip(group_list,fake_group_list):
    insert_dict={'group_name':group,
                'fake_group_name':fake_group}
    rows_list.append(insert_dict)
fake_group_df = pd.DataFrame(rows_list)   
```

Now we initialize a dataframe called __gf__ which is the entries from the original database without duplication on the company level.  

This dataframe will be our group and company mapping dataframe, or anonymizer.  We remove all duplicates so that gf now has a 1 - many relationship between group and company and the company column contains unique names only.  

Now we merge the fake_company_df dataframe onto gf so that there is a new column for our fake company names.  The merge result has to have as many rows as gf!  

After this we merge the group dataframe on to this result which then adds the fake group name onto our mapping dataframe.  

Once again the final dataframe has to have the same number of rows as the original gf:

```python
gf=df[['company','group']].copy()
gf.drop_duplicates(inplace=True)
gf=gf.merge(fake_company_df, left_on='company', right_on='company_name', how='left')
gf=gf.merge(fake_group_df, left_on='group', right_on='group_name',how='left')
```

Now merge the anonymizer dataframe back onto df and preserve the original dataframe.  This adds four columns 
- 'company_name',
- 'fake_company_name',
- 'group_name',
- 'fake_group_name'

```python
hf=df.merge(gf[['company_name',
 'fake_company_name',
 'group_name',
 'fake_group_name']], left_on='company', right_on='company_name', how='left')
```

See the changes in the shape of the two data frames, notice that the number of rows did not change and that there are now 4 more columns (as it should be):

```python
print('Shape of the original dataframe: ', df.shape, '; shape of the new dataframe: ', hf.shape)
```

    Shape of the original dataframe:  (1223334, 52) ; shape of the new dataframe:  (1223334, 56)


We can get rid of the columns
- 'company_name',
- 'company',
- 'group',
- 'group_name,
in the mapping dataframe, after which we can rename __fake_company_name__ to __company__ and __fake_group_name to group to complete the process.

```python
hf=df.merge(gf[['company_name',
 'fake_company_name',
 'group_name',
 'fake_group_name']], left_on='company', right_on='company_name', how='left')
hf=hf.drop(['company_name',
'company',
'group_name',
'group'], axis=1)
hf.rename(columns={'fake_company_name':'company',
          'fake_group_name':'group'}, inplace=True)
```

Verify that the shape of the original file shapes have not changed!

```python
print(hf.shape, df.shape)
```

    (1223334, 52) (1223334, 52)


This is it for the company and group anonymization.  We have successfully anonymized two relevant and useful columns for future use!

## Member names

Personal names are found in the __member__ and __individual__ columns.  

Members are the primary individual, and other individuals are the dependents.  In the anonymization, members will be given a unique surname, shared with all the dependents, but only the dependents.  We will anonymize individuals further with initials, to make sure no two individuals share the same surname initials combination.  

There is also a gender column that is for future use.

We create a member and individual list each containing unique numbers:

```python
member_list=list(set(df['member']))
individual_list=list(set(df['individual']))
```

There are 1,223,334 records and 53028 individual unique members:

```python
len(df['member']),len(set(df['member']))
```




    (1223334, 53028)



Likewise there are 75,396 individuals:

```python
len(df['member']),len(set(df['individual']))
```




    (1223334, 75395)



Lets create a dataframe with only unique individuals and there corresponding member, keep in mind there is a one to many relationship between member and individual, the member columns will contain duplicates:

```python
cf=df[['member','individual']].drop_duplicates()
```

In the following we extract all the names from the __NameDatasetV1__ database.  Then, extract the surnames from these tuples and eliminate surnames that contain non standard characters.  With this we can index into a fake name list to have a mapping between real and fake surnames:

```python
names = NameDatasetV1()
last_names_list=list(set(list(names.last_names)))
alpha_last_names_idx = list(np.where([str1.isalpha() for str1 in last_names_list])[0])
last_names_list = [last_names_list[idx] for idx in alpha_last_names_idx]
surname_list=last_names_list[0:len(member_list)]
```

```python
print('length of surname_list: ', len(surname_list), 'length of the fake surname list, last_names_list: ',len(last_names_list))
```

    length of surname_list:  53028 length of the fake surname list, last_names_list:  97022


In the cf data frame we now create a surname column containing a unique family surname for each member (and their dependent).

```python
for row in cf.iterrows():
    member=row[1]['member']
#     print(row[1]['member'],row[1]['individual'])
#     print(surname_list[member_list.index(row[1]['member'])])
    surname=surname_list[member_list.index(row[1]['member'])]
    cf.loc[cf.member==member,'surname']=surname.capitalize()
```


There are still duplicates as the following reveals.

The following cell adds initials to a surname and store the result in the name column, the number of initials is rnadomized between 1 and 3.

```python
for row in cf.iterrows():
    name=row[1]['surname']
    individual=row[1]['individual']
#     name[0]=name[0].ucase()
#     print(row[1]['member'],row[1]['individual'])
#     print(surname_list[member_list.index(row[1]['member'])])
    a=int(random()*25)
    b=int(random()*25)
    c=int(random()*25)
    d=random()
    if d<=0.1:
        s=alpha[a]+'. '+alpha[b]+'. '+alpha[c]+'. '+name        
    if d<=0.3 and d>0.1:
        s=alpha[a]+'. '+alpha[b]+'. '+name       
    if d>0.3:
        s=alpha[a]+'. '+name              
#     s=alpha[a]+'.'+alpha[b]+'. '+name
    print(s)
#     surname=surname_list[member_list.index(row[1]['member'])]
    cf.loc[cf.individual==individual,'name']=s
```

In this cell we remove duplicates by adding another initial, in case there are instances where only 1 initial was added and the resulted in the same name for two family members!

```python
cf_name_duplicated=cf[cf.name.duplicated()].copy()
for row in cf_name_duplicated.iterrows():
    
    name=row[1]['name']
    print(name)
    individual=row[1]['individual']
    a=int(random()*25)
    s=alpha[a]+'. '+name         
    cf.loc[cf.individual==individual,'name']=s    
```

    K. Emmert
    Z. Delpozo
    B. Boname
    S. Camellon
    O. Weaving
    C. Heft
    L. Benjamine
    Y. Leviner
    ...

Now we can see that there are no duplications:

```python
len(cf.name),len(set(cf.name))
```


    (75395, 75395)



<!-- Save the anonynimazation map: -->

And that is it!  This has now anonymized the member list and their dependent list into a mapping file for future use.

## Provider

We now turn our attention to the provider details.  There are 8,184 unique providers:

```python
provider_list=list(set(df['provider']))
len(df['provider']),len(set(df['provider']))
```




    (1223334, 8184)



This is where a little of the magic happens.  We read from NameDatasetV1 unique surnames to have one for each of our providers.  We are mostly interested in the __provider_surname_list__.  This is a unique list of names from which each provider can be uniquely identified:

```python
names = NameDatasetV1()
last_names_list=list(set(list(names.last_names)))
alpha_last_names_idx = list(np.where([str1.isalpha() for str1 in last_names_list])[0])
last_names_list = [last_names_list[idx] for idx in alpha_last_names_idx]
provider_surname_list=last_names_list[(len(last_names_list)-len(provider_list)-1):-1]
```

Verify that the provider_surname_list contains a list of unique names:

```python
len(provider_surname_list),len(set(provider_surname_list)), len(provider_list), len(set(provider_list))
```

    (8184, 8184, 8184, 8184)



Once again we create a mapping dataframe from the original dataframe:

```python
cf=df[['provider','pr_type_descr']].drop_duplicates()
```

This shows there are doctors connected to more than one provider type!:

```python
len(set(cf.provider)),len(list(cf.provider))
```




    (8184, 8197)



In the following we choose a surname (without replacement) for each __provider__ from a list of unique surnames.  If the provider number was 
used for two different practice types, we will still attach the same surname to these practices with the logic below:

```python
for row in cf.iterrows():
    provider=row[1]['provider']
#     print(row[1]['member'],row[1]['individual'])
#     print(surname_list[member_list.index(row[1]['member'])])
    provider_surname=provider_surname_list[provider_list.index(provider)]
    cf.loc[cf.provider==provider,'provider_surname']=provider_surname.capitalize()
```

In the below cell we pad each surname with random characters (token initials) to further personalize the names and make these even more believable and as said before, easier on the human eye:

```python
for row in cf.iterrows():
    name=row[1]['provider_surname']
    provider=row[1]['provider']
#     name[0]=name[0].ucase()
#     print(row[1]['member'],row[1]['individual'])
#     print(surname_list[member_list.index(row[1]['member'])])
    a=int(random()*25)
    b=int(random()*25)
    c=int(random()*25)
    d=random()
    if d<=0.1:
        s=alpha[a]+'. '+alpha[b]+'. '+alpha[c]+'. '+name        
    if d<=0.3 and d>0.1:
        s=alpha[a]+'. '+alpha[b]+'. '+name       
    if d>0.3:
        s=alpha[a]+'. '+name              
#     s=alpha[a]+'.'+alpha[b]+'. '+name
    print(s)
#     surname=surname_list[member_list.index(row[1]['member'])]
    cf.loc[cf.provider==provider,'provider_name']='Dr. '+s
```

    S. X. Chotibai
    P. Langenbach
    P. Belone
    M. B. Arban
    M. T. Zapato
    C. Beecken
    P. Newens
    X. Agresto
    T. Allgood
    ...

```python
# cf.to_csv(path_or_buf="""/notebooks/github_repos/fraud/data/provider_map.csv""", index=False)
```

## Putting it all together

In this section the mapping dataframes are used to encode the original dataset and save it for further use.

```python
list(company_group_map_df)
```




    ['company',
     'group',
     'company_name',
     'fake_company_name',
     'group_name',
     'fake_group_name']



```python
anon_df=df.merge(company_group_map_df[['company',
 'fake_company_name',
 'fake_group_name']], right_on='company',left_on='company',how='left')
```

```python
print(df.shape, anon_df.shape)
```

    (1223334, 52) (1223334, 54)


```python
list(member_map_df)
```




    ['member',
     'individual',
     'fake_surname',
     'fake_name',
     'member_code',
     'fake_member']



```python
anon_df=anon_df.merge(member_map_df[['individual', 'fake_member','fake_surname', 'fake_name']], 
                      left_on='individual',
                     right_on='individual',
                     how='left')
```

```python
anon_df[[
    'fake_member',
    'fake_company_name',
    'fake_group_name',
    'fake_surname',
    'fake_name'
]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fake_member</th>
      <th>fake_company_name</th>
      <th>fake_group_name</th>
      <th>fake_surname</th>
      <th>fake_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L. Soravilla</td>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L. Soravilla</td>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
    </tr>
    <tr>
      <th>2</th>
      <td>L. Soravilla</td>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
    </tr>
    <tr>
      <th>3</th>
      <td>L. Soravilla</td>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L. Soravilla</td>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1223329</th>
      <td>V. Turturo</td>
      <td>Electronic Technology</td>
      <td>Solutions Building Net</td>
      <td>Turturo</td>
      <td>V. Turturo</td>
    </tr>
    <tr>
      <th>1223330</th>
      <td>E. X. W. Mccardle</td>
      <td>Alpha Studio North</td>
      <td>Solutions Building Net</td>
      <td>Mccardle</td>
      <td>A. B. W. Mccardle</td>
    </tr>
    <tr>
      <th>1223331</th>
      <td>V. Turturo</td>
      <td>Electronic Technology</td>
      <td>Solutions Building Net</td>
      <td>Turturo</td>
      <td>V. Turturo</td>
    </tr>
    <tr>
      <th>1223332</th>
      <td>V. Turturo</td>
      <td>Electronic Technology</td>
      <td>Solutions Building Net</td>
      <td>Turturo</td>
      <td>V. Turturo</td>
    </tr>
    <tr>
      <th>1223333</th>
      <td>L. Parrsr</td>
      <td>Frontier Galaxy Architecture</td>
      <td>Solutions Building Net</td>
      <td>Parrsr</td>
      <td>L. Parrsr</td>
    </tr>
  </tbody>
</table>
<p>1223334 rows × 5 columns</p>
</div>



Look at the list of columns in provider_map_df:

```python
list(provider_map_df)
```




    ['provider', 'pr_type_descr', 'fake_provider_surname', 'fake_provider_name']



```python
anon_df=anon_df.merge(provider_map_df[['provider', 'fake_provider_surname', 'fake_provider_name']], 
                      left_on='provider',
                     right_on='provider',
                     how='left')
```

This then is what the anonymized section of our dataset looks like.  Each one of the rows below is a single claim event.

```python
anon_df[[ 'fake_company_name',
         'fake_group_name',
        'fake_surname', 
         'fake_name','fake_member',
        'fake_provider_surname', 
         'fake_provider_name']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fake_company_name</th>
      <th>fake_group_name</th>
      <th>fake_surname</th>
      <th>fake_name</th>
      <th>fake_member</th>
      <th>fake_provider_surname</th>
      <th>fake_provider_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
      <td>L. Soravilla</td>
      <td>Chotibai</td>
      <td>Dr. S. X. Chotibai</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
      <td>L. Soravilla</td>
      <td>Chotibai</td>
      <td>Dr. S. X. Chotibai</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
      <td>L. Soravilla</td>
      <td>Chotibai</td>
      <td>Dr. S. X. Chotibai</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
      <td>L. Soravilla</td>
      <td>Chotibai</td>
      <td>Dr. S. X. Chotibai</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Digital Virtual</td>
      <td>Construction East Frontier</td>
      <td>Soravilla</td>
      <td>L. Soravilla</td>
      <td>L. Soravilla</td>
      <td>Chotibai</td>
      <td>Dr. S. X. Chotibai</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1226345</th>
      <td>Electronic Technology</td>
      <td>Solutions Building Net</td>
      <td>Turturo</td>
      <td>V. Turturo</td>
      <td>V. Turturo</td>
      <td>Shopbell</td>
      <td>Dr. L. Shopbell</td>
    </tr>
    <tr>
      <th>1226346</th>
      <td>Alpha Studio North</td>
      <td>Solutions Building Net</td>
      <td>Mccardle</td>
      <td>A. B. W. Mccardle</td>
      <td>E. X. W. Mccardle</td>
      <td>Boid</td>
      <td>Dr. Y. Boid</td>
    </tr>
    <tr>
      <th>1226347</th>
      <td>Electronic Technology</td>
      <td>Solutions Building Net</td>
      <td>Turturo</td>
      <td>V. Turturo</td>
      <td>V. Turturo</td>
      <td>Shopbell</td>
      <td>Dr. L. Shopbell</td>
    </tr>
    <tr>
      <th>1226348</th>
      <td>Electronic Technology</td>
      <td>Solutions Building Net</td>
      <td>Turturo</td>
      <td>V. Turturo</td>
      <td>V. Turturo</td>
      <td>Shopbell</td>
      <td>Dr. L. Shopbell</td>
    </tr>
    <tr>
      <th>1226349</th>
      <td>Frontier Galaxy Architecture</td>
      <td>Solutions Building Net</td>
      <td>Parrsr</td>
      <td>L. Parrsr</td>
      <td>L. Parrsr</td>
      <td>Frothingham</td>
      <td>Dr. G. Y. O. Frothingham</td>
    </tr>
  </tbody>
</table>
<p>1226350 rows × 7 columns</p>
</div>



By choosing only relevant columns including those that were anonymized, in a list called anon_column_list, we can now save the anon_df dataset ready for further use:

```python
anon_df[anon_column_list].to_csv(path_or_buf="""anon_df.csv""", index=False)
```

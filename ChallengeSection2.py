#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# In[2]:


data_2017 = pd.read_csv('PartD_Prescriber_PUF_NPI_17/PartD_Prescriber_PUF_NPI_17.txt', delimiter='\t')


# Note: data has 84 columns while the documentation lists 89, but numbers 64-68 are missing in the documentation.

# ### Question 1
# 
# **In 2017, what was the average number of beneficiaries per provider? Due to the suppression of data for those with few beneficiaries, we can only include those with more than 10 beneficiaries.**
# 
# The `bene_count` column contains information on the number of beneficiaries per provider. Therefore, we can inspect the values to find the average number of beneficiaries per provider.

# In[3]:


bene_count_data = data_2017['bene_count'].values[~np.isnan(data_2017['bene_count'].values)]


# By dropping the nan values, we are only counting the providers with more than 10 beneficiaries. This is verified by checking the minimum value in the array:

# In[4]:


print(bene_count_data.min())


# In[5]:


print(bene_count_data.sum() / len(bene_count_data))


# ## Question 2
# 
# **For each provider, estimate the length of the average prescription from the total_day_supply and total_claim_count. What is the median, in days, of the distribution of this value across all providers?**

# In[6]:


average_perscription_data = data_2017['total_day_supply'].values / data_2017['total_claim_count'].values
print(np.median(average_perscription_data))


# Note: the median of this distribution is in units of days per claim.

# ## Question 3
# 
# **Work out for each Specialty the fraction of drug claims that are for brand-name drugs. Include only providers for whom the relevant information has not been suppressed, and consider only specialties with at least 1000 total claims. What is the standard deviation of these fractions?**

# In[7]:


specialties = np.unique(data_2017['specialty_description'].values)
fractions = []
for specialty in specialties:
    specialty_data = data_2017[data_2017['specialty_description'].values == specialty].copy()
    
    # Only include providers for whom the relevant information has not been suppressed
    specialty_data = specialty_data[(specialty_data['brand_suppress_flag'].values != '#') &
                                    (specialty_data['brand_suppress_flag'].values != '*')].copy()
        
    # Only consider specialties with at least 1000 total claims
    total_claims = specialty_data['total_claim_count'].values.sum()
    if total_claims < 1000:
        continue
    
    brand_name_claims = specialty_data['brand_claim_count'].values.sum()
    fractions.append(brand_name_claims / total_claims)
    
print(np.std(fractions))
    


# ## Question 4
# 
# **Find the ratio of beneficiaries with opioid prescriptions to beneficiaries with antibiotics prescriptions in each state. Assume that each beneficiary attends only a single provider. What is the difference between the largest and smallest ratios?**

# In[8]:


state_dict = {}
unique_states = np.unique(data_2017['nppes_provider_state'].values)

state_values_to_exclude = ["XX", "AA", "AE", "AP", "AS", 
                           "GU", "MP", "PR", "VI", "ZZ"]

for state in unique_states:
    
    if state in state_values_to_exclude: continue
        
    state_data = data_2017[data_2017['nppes_provider_state'].values == state].copy()
    
    num_opiod = state_data['opioid_bene_count'].values[~np.isnan(state_data['opioid_bene_count'].values)].sum()
    num_antiboitic = state_data['antibiotic_bene_count'].values[~np.isnan(state_data['antibiotic_bene_count'].values)].sum()
    
    state_dict[state] = num_opiod / num_antiboitic
    
print(np.max(list(state_dict.values())) - np.min(list(state_dict.values())))


# ## Question 5
# 
# **For each provider where the relevant columns are not suppressed, work out the fraction of claims for beneficiaries age 65 and older, as well as the fraction of claims for beneficiaries with a low-income subsidy. What is the Pearson correlation coefficient between these values?**

# In[9]:


#Get data where 65+ and low-income are both not suppressed
data = data_2017[(data_2017['ge65_suppress_flag'].values != '*') &
                 (data_2017['ge65_suppress_flag'].values != '#') &
                 (data_2017['lis_suppress_flag'].values != '*') &
                 (data_2017['lis_suppress_flag'].values != '#')].copy()

ge65_fractions = data['total_claim_count_ge65'].values / data['total_claim_count'].values
lis_fractions = data['lis_claim_count'].values / data['total_claim_count'].values

print(pearsonr(ge65_fractions, lis_fractions)[0])


# ## Question 6
# 
# **Let's find which states have surprisingly high supply of opioids, conditioned on specialty. Work out the average length of an opioid prescription for each provider. For each (state, specialty) pair with at least 100 providers, calculate the average of this value across all providers. Then find the ratio of this value to an equivalent quantity calculated from providers in each specialty across all states. What is the largest such ratio?**

# In[10]:


state_values_to_exclude = ["XX", "AA", "AE", "AP", "AS", 
                           "GU", "MP", "PR", "VI", "ZZ"]

# Group by specialty and state
grouped_data_2017 = list(data_2017.groupby(['specialty_description', 'nppes_provider_state']))
grouped_data_2017 = [x for x in grouped_data_2017 if x[0][1] not in state_values_to_exclude]

# Require at least 100 providers in the pair for it to count
grouped_data_2017 = [x for x in grouped_data_2017 if x[1].shape[0] >= 100]


# In[11]:


specialty_dict = {}
for (specialty, state), data in grouped_data_2017:
    
    opioid_perscription_data = data['opioid_day_supply'].values
    opioid_perscription_data = opioid_perscription_data[~np.isnan(opioid_perscription_data)]
    
    if specialty in specialty_dict.keys():
        specialty_dict[specialty]['SUPPLY'] += [opioid_perscription_data.sum()]
        specialty_dict[specialty]['NUM'] += [len(opioid_perscription_data)]
    else:
        specialty_dict[specialty] = {}
        specialty_dict[specialty]['SUPPLY'] = [opioid_perscription_data.sum()]
        specialty_dict[specialty]['NUM'] = [len(opioid_perscription_data)]


# In[12]:


state_specialty_averages = {k: [x / y for x, y in zip(v['SUPPLY'], v['NUM'])] for k, v in specialty_dict.items()}
all_state_averages = {k: np.sum(v['SUPPLY']) / np.sum(v['NUM']) for k, v in specialty_dict.items()}

largest_ratio = 0.0
for specialty in all_state_averages.keys():
    new_ratio = np.max(state_specialty_averages[specialty]) / all_state_averages[specialty]
    if new_ratio > largest_ratio:
        largest_ratio = new_ratio


# In[13]:


largest_ratio


# ## Question 7
# 
# **For each provider for whom the information is not suppressed, figure out the average cost per day of prescriptions in both 2016 and 2017. Use this to estimate the inflation rate for daily prescription costs per provider. What is the average inflation rate across all providers?**

# In[14]:


data_2016 = pd.read_csv('PartD_Prescriber_PUF_NPI_16/PartD_Prescriber_PUF_NPI_16.txt', delimiter='\t')


# In[15]:


data_2016['cost_per_day_2016'] = data_2016['total_drug_cost'].values / data_2016['total_day_supply'].values
data_2017['cost_per_day_2017'] = data_2017['total_drug_cost'].values / data_2017['total_day_supply'].values


# In[16]:


merged_data = data_2016.merge(data_2017, on='npi')


# In[17]:


cost_changes = merged_data['cost_per_day_2017'].values - merged_data['cost_per_day_2016'].values


# In[18]:


cost_changes = cost_changes[~np.isnan(cost_changes)]


# In[19]:


print(np.mean(cost_changes) / 365)


# ## Question 8
# 
# **Consider all providers with a defined specialty in both years. Find the fraction of providers who left each specialty between 2016 and 2017. What is the largest such fraction, when considering specialties with at least 1000 proviers in 2016? Note that some specialties have a fraction of 1 due to specialty name changes between 2016 and 2017; disregard these specialties in calculating your answer.**

# In[20]:


data_2016['2016_specialty'] = data_2016['specialty_description'].values
data_2017['2017_specialty'] = data_2017['specialty_description'].values

merged_data = data_2016.merge(data_2017, on='npi')


# In[21]:


specialties_2016, counts_2016 = np.unique(merged_data['2016_specialty'].values, return_counts=True)
popular_specialties = specialties_2016[counts_2016 > 1000]


# In[22]:


fractions = []
for specialty in popular_specialties:
    
    specialty_df = merged_data[merged_data['2016_specialty'].values == specialty].copy()
    
    fractions.append(np.sum((specialty_df['2017_specialty'].values != specialty)) / specialty_df.shape[0])


# In[23]:


# Disregard cases where fraction == 1.0
fractions = [x for x in fractions if x != 1.0]
print(np.max(fractions))


# In[ ]:





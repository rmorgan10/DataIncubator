#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random


# In[2]:


def possible_coin_values(N):
    """
    Generate a list of all possible coin values to draw from
    
    :param N: (positive int) number of coins to draw
    :return: coin_values: (list) numbers from 1 to N, inclusive
    """
        
    return list(np.arange(N) + 1)

def draw_coins(coin_values):
    """
    Drawing all coins without replacement is equivalent to 
    shuffling the list. This function shuffles the list.
    
    :param: coin_values: output of possible_coin_values
    :return: drawn_coins: coin values in the order they were drawn
    """
    random.shuffle(coin_values)
    
    return coin_values

def calculate_payment(drawn_coins):
    """
    Calculate the total payment for drawn coins
    
    :param drawn_coins: output of draw_coins
    :return: payment: a single realization of drawing from coin_values
    """   
    return np.sum([drawn_coins[0]] + [np.abs(drawn_coins[i + 1] - drawn_coins[i]) for i in range(len(drawn_coins) -1)])

def perform_realization(N):
    """
    Draw N coins and calculate the payment
    
    :param N: (positive int) number of coins to draw
    :return: payment: calculated payment
    """
    if type(N) != int or N <= 0:
        raise  Exception("argument must be positive integer")
        
    coin_values = possible_coin_values(N)
    drawn_coins = draw_coins(coin_values)
    return calculate_payment(drawn_coins)


# In[3]:


# N = 10 payments
size = 1000000
n10_payments = [perform_realization(10) for _ in range(size)]


# In[4]:


print(np.mean(n10_payments))
print(np.std(n10_payments))


# In[5]:


# N = 20 payments
size = 1000000
n20_payments = [perform_realization(20) for _ in range(size)]


# In[6]:


print(np.mean(n20_payments))
print(np.std(n20_payments))


# In[7]:


n10_array = np.array(n10_payments)
n20_array = np.array(n20_payments)


# In[8]:


print(np.sum((n10_array > 45.0)) / len(n10_array))


# In[9]:


print(np.sum((n20_array > 160.0)) / len(n20_array))


# In[ ]:





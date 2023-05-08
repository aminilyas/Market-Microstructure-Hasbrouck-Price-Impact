# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 16:13:10 2022

@author: Group 2
1. Amin ILYAS
2. Nico Benedikt HORSTMANN
3. Nizar AQACHMAR 
4. Pritam RITU RAJ
5. Zahi SAMAHA
"""
####################################################################################
#                                   PROJECT TITLE
#                               HASBROUCK PRICE IMPACT
####################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

####################################################################################
#   1 Select one corporation among the dataset provided by the professor
####################################################################################
print("The selected corporation is General Electric.")

df = pd.read_excel("GE_LOB_ExecuteOrder_10042021.xlsx")
print(df)


####################################################################################
#   2 Merge the trade file with the quote file
####################################################################################
#Step 2 is not performed because the quote file is not available.


####################################################################################
#   3 For each trading day:
#   - Split the trading into XX buckets of 5 minutes (X≈70)
#   - Compute 5-minute returns and signed volume
#   - Do a linear regression to estimate the Hasbrouck Price Impact measure
####################################################################################

#=============3.1 Split the trading into XX buckets of 5 minutes (X≈70)============#

#   3.1.1 Buy and sell volume trade classification
df['tick'] = np.nan
for i in range(1, len(df)):
    if df.iloc[i,6] > df.iloc[i-1,6]:
        df.iloc[i,-1] = 1
    elif df.iloc[i,6] < df.iloc[i-1,6]:
        df.iloc[i,-1] = -1
    else:
        df.iloc[i,-1] =  df.iloc[i-1,-1]
print(df)

#   3.1.2 signed square root dollar volume
df['vol'] = df['tick']*(df['Volume']**(1/2))

#   3.1.3 Convert time to str to set the times into the same format
df['SourceTime'] = df['SourceTime'].astype(str)
# Adding milliseconds to time with no ms
for i in range(0, len(df)):
    if len(df.iloc[i,3]) < 15:
        df.iloc[i,3] = df.iloc[i,3] +'.000000'
df['SourceTime']= pd.to_datetime(df['SourceTime'], format='%H:%M:%S.%f')
df = df.set_index(['SourceTime'])

#   3.1.4 Dataframe for price
price = df[['Price']]
price = price.resample('5min').median()
price = price.reset_index(drop=False)
price['SourceTime']= pd.to_datetime(price['SourceTime'], format='%H:%M:%S.%f').dt.time

#   3.1.5 Dataframe for volume
vol = df[['vol']]
vol = vol.resample('5min').sum()
vol = vol.reset_index(drop=False)
vol['SourceTime']= pd.to_datetime(vol['SourceTime'], format='%H:%M:%S.%f').dt.time

#   3.1.6 Merge price and volume dataframes
data = pd.merge(price, vol, on='SourceTime')

#==============3.2 Compute 5-minute returns and signed volume======================#

#   3.2.1 Compute price return
data['return'] = np.nan
for i in range(1,len(data)):
    data.iloc[i,3] = np.log(data.iloc[i,1]/data.iloc[i-1,1])

#   3.2.2 Remove extreme values
data.iloc[-1,2] = -1*((-1*data.iloc[-1,2])**(1/2))

#   3.2.3 Modify the scale of return
data['return'] = data['return']*100

#   3.2.4 Drop nan values and reset index
data.dropna(inplace= True)
data = data.reset_index(drop=True)


#=====3.3 Do a linear regression to estimate the Hasbrouck Price Impact measure====#

#   3.3.1 Define linear regression model:
slr="Hasbrouck's price impact equation is defined as Rn = λ x Sn + En.\n"\
"Where: \n"\
"Rn is stock return over a 5-minute interval \n"\
"Sn is signed square root dollar volume over a 5-minute interval \n"\
"En is error term and \n"\
"λ is dynamic price impact measure estimated by OLS N ≈ 200."
print(slr)

#   3.3.2 Variables: Sn and Rn
data['Sn'] = data['vol']
data['Rn'] = data['return']

    
#   3.3.3 Lambda (λ)
def estimate_coef(df, x, y):
   features = [x]
   target = y
   X = df[features].values.reshape(-1, len(features))
   y = df[target].values
   ols = linear_model.LinearRegression(fit_intercept=False)
   model= ols.fit(X, y)
   return  model.coef_[0]

λ = estimate_coef(data,'Sn', 'Rn')

print(λ)
print("Estimated coefficients:")
print("Slope (λ) = ", λ)

#   3.3.4 Plot linear regression (Hasbrouck equation)
sns.regplot(data['Sn'], data['Rn'])
plt.legend(title='Regression model')
plt.show()

####################################################################################
#   4 Provide a graphical representation on the price impact dynamic through time
####################################################################################

#   4.1 Compute lambda for every two consecutive SourceTime
l = []
for i in range(0, len(data)-1):
    data2 = data.iloc[i:i+2, 2]
    data2 = pd.DataFrame(data2)
    data2['return'] = data.iloc[i:i+2,3]
    a = estimate_coef(data2,'vol', 'return')
    l.append(a)
    
data2 = pd.DataFrame(l)
data2 = data2.rename(columns = {0:'lambda'})
data2['SourceTime'] = data.iloc[0:77, 0]

#   4.2 PLot Price imact dynamic through time
data2.plot(x='SourceTime', y='lambda')
plt.grid()
plt.xlabel('Time')
plt.ylabel('Lambda')

#   4.3 Intepretation of plot in 4.2
"""
As shown by the plot, the values of lambda varies with time. 
The variation is due to the multifactors including changes in stock return and volume of dollar over time.
"""


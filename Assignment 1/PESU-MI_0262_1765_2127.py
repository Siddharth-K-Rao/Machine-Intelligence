'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
	entropy = 0
	values = df[df.columns[-1]].unique()
	for i in values:
		fract_val = df[df.columns[-1]].value_counts()[i]/len(df[df.columns[-1]])
		entropy += -fract_val*np.log2(fract_val)
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	values = df[df.columns[-1]].unique()
	features = df[attribute].unique()
	entropy_of_attribute = 0

	for i in features:
		entropy_feature = 0
		den = len(df[attribute][df[attribute]==i])
		for j in values:
			num = len(df[attribute][df[attribute]==i][df[df.columns[-1]]==j])
			#den = len(df[attribute][df[attribute]==i])
			fract = float(num/den)
			if(fract != 0):
				entropy_feature += -fract*np.log2(fract)
			else:
				entropy_feature += -fract*np.log2(10**-7)
		fract_val = den/len(df)
		entropy_of_attribute += -fract_val*entropy_feature

	return abs(entropy_of_attribute)

'''def average_information(df):
	entropy = {i:get_entropy_of_attribute(df, i) for i in df.keys()[:-1]}
	return average_information'''

'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = 0
	information_gain = (get_entropy_of_dataset(df) - get_entropy_of_attribute(df, attribute))

	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''

	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''
	information_gains = {i:get_information_gain(df, i) for i in df.keys()[:-1]}
	selected_column = max(information_gains, key=information_gains.get)

	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''

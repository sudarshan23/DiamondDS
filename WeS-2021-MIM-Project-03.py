# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 19:09:47 2020

Semester V, Advanced Business Analytics.

dataset: diamonds-m.csv
project: Project Work as per 202012-Diamonds-Project.pdf

    @authors-           Roll numbers-
    Sudarshan Chokhani  07
"""

########################################################################################
# Importing all the required libraries and packages.
########################################################################################

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# import modules

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import seaborn as sns
import utils

########################################################################################
# SECTION - 1
# Read & Exploratory Data Analysis
# - Read the data.
# - Basic exploration and inferences about the structure.
# - Display Average Price in Crosstab with Carat and Cut.
########################################################################################

#### 1.a Read the data set.
df = pd.read_csv('diamonds-m.csv')

#### 1.b Show the structure and basic summary
# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())
print(df.shape)

## (53940, 12)

# summary
print("\n*** Summary ***")
print(df.describe())

"""
Output:
    # 		id				carat			depth			table			price			x				y				z
    # count	53940.000000 	53938.000000 	53940.000000 	53940.000000	    53936.000000	    53940.000000	    53940.000000	    53940.000000
    # mean	26970.500000 	0.797947		 	61.749405		57.457184		3932.833822		5.731157		    5.734526			3.538734
    # std	15571.281097	    0.474018		 	1.432621		 	2.234491			3989.443555		1.121761		    1.142135			0.705699
    # min	1.000000		    0.200000		 	43.000000		43.000000		326.000000		0.000000		    0.000000		    0.000000
    # 25%	13485.750000    	0.400000		 	61.000000		56.000000		950.000000		4.710000		    4.720000		    2.910000
    # 50%	26970.500000	    0.700000		 	61.800000		57.000000		2401.000000		5.700000		    5.710000		    3.530000
    # 75%	40455.250000	    1.040000		 	62.500000		59.000000		5324.250000		6.540000		    6.540000		    4.040000
    # max	53940.000000    	5.010000		 	79.000000		95.000000		18823.000000		10.740000	    58.900000	    31.800000


Initial Observations for numerical variables.:
    
# 1. ID column doesnt have any meaning to the dataset ,we will remove it.
# 2. There are zero's(0) in x,y & z columns, doesnt make any sense for height width and length of diamond to be zero.
# 3. Price is in defined range of $326 to $18,823.
# 4. Carat is in defined range of 0.2 to 5.01.
# 5. x, y, z (mm) are in defined ranges respectively.
# 6. 1 carat is 200mg, might come in handy with feature engineering.
    
"""

#### Display Average Price in Crosstab with Carat & Cut
#### Filled the NaN values with zeros.
print(pd.crosstab(df.carat, df.cut, values=df.price, aggfunc='mean', margins=True, margins_name="Total").fillna(0))

####  Cross Tabs as seperate tables.
df.groupby(['cut'])['price'].mean()
df.groupby(['carat'])['price'].mean()

#### Check For Zeros In Numeric Columns (except column “Price”). Convert to Null.
# We observer int64 and float64 as numeric datatypes
# Disntinguishing Numberical from categorial features.

########################################################################################
## Section - 2
## Data cleaning
## Handling Data
########################################################################################

########################################################################################
## Data cleaning - Numerical features
########################################################################################

## id column is identifier column with no relevance to the dataset.
## we will drop it from the data set

df=df.drop(['id'],axis = 1)

# list of numerical variables
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))

#remove the price feature
numerical_features.remove('price')

# visualise the numerical variables
df[numerical_features].head()

# Initial value of zeros in numerical features
print((df[numerical_features]==0).sum())

"""
Output:
    Number of numerical variables:  7
    carat     0
    depth     0
    table     0
    x         8
    y         7
    z        20
    dtype: int64

 Observation:
    x,y,z columns which are related to dimensions of diamond has 8,7 & 20 values of zero(0), total - 35 values
    Replace the zero's with NAN and display the counts again. 
    Doesnt make any sense for the physcial attributes to be zero.
"""

df[numerical_features]=df[numerical_features].replace(0, np.nan)

# After replacement validaiton of data.
print((df[numerical_features]==0).sum())

"""
Output:
    carat    0
    depth    0
    table    0
    x        0
    y        0
    z        0
    dtype: int64
"""

# there are no zeros now in the numberical features.

########################################################################################
## Data cleaning - Categorical features
########################################################################################

# list of categorical variables
categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
print('Number of categorical variables: ', len(categorical_features))

# Number of categorical variables:  4

#remove the cut feature - which is the class feature later in our classification alogrithm.
categorical_features.remove('cut')
# visualise the categorical variables
df[categorical_features].head()

"""
Output:
     color clarity popularity
    0     E     SI2       Good
    1     E     SI1       Good
    2     E     VS1       Fair
    3     I     VS2       Poor
    4     J     SI2       Good
"""

# get unique Class names
print("\n*** Unique Class - Categoric ***")
for column in categorical_features:
     print("Column: ",column, "    Values: " ,df[column].unique())

"""
Output:
    
*** Unique Class - Categoric ***
Column:  color     Values:  ['E' 'I' 'J' 'H' 'F' 'G' 'D' nan]
Column:  clarity     Values:  ['SI2' 'SI1' 'VS1' 'VS2' 'VVS2' 'I1' 'VVS1' 'IF']
Column:  popularity     Values:  ['Good' 'Fair' 'Poor' nan 'NotAvail']

Observation:
    
    We see invalid entries in the color as NaN and popularity as nan & "NotAvail".
"""

# we see invalid entries in the color as NaN and popularity as nan & NotAvail

########################################################################################
## Data cleaning - and NAN value Imputation
########################################################################################

# Initial count of entries as 'NotAvail' in popularity.
print("Invalid entries in column : Popularity")
print((df['popularity']=='NotAvail').sum())

"""
Output:
    Invalid entries in column : Popularity
    13
    
Observation:
    Total number of records still less than 1% of total dataset, so we converted to NaN. And will drop it later.
"""

print("As a quick fix, converting these values to NaN")
df['popularity']=df['popularity'].replace('NotAvail', np.nan)

### Validating the data
print("Invalid entries in column : Popularity")
print((df['popularity']=='NotAvail').sum())

"""
Output:
    Invalid entries in column : Popularity
    0
Observation:
    Invalid label entries converted to NaN.
"""

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

print('\n*** Categorical Columns With Nulls ***')
print(df[categorical_features].isnull().sum()) 

print('\n*** Numerical Columns With Nulls ***')
print(df[numerical_features].isnull().sum())

"""
Output:
    
*** Columns With Nulls ***
carat          2
cut            0
color          3
clarity        0
popularity    23
depth          0
table          0
price          4
x              8
y              7
z             20
dtype: int64

*** Categorical Columns With Nulls ***
color          3
clarity        0
popularity    23
dtype: int64

*** Numerical Columns With Nulls ***
carat     2
depth     0
table     0
x         8
y         7
z        20
dtype: int64

Observation/Reasoning:
    1. Till now we have converted, invalid zeros's(0) and invaid categories to nan values.
    2. There are total of 63 null values(55 records) from 54940 records till now. -- less than 1% of data
    3. We will go ahead and safely drop this records from the dataset.
    4. 
"""
 
# segregating the prediction data seperatly to a price oriented dataframe called dfprice
# safely 4 records are present in the dfprice dataframe
    
dfprice = df[df['price'].isnull()]
dfprice.head()

"""
Output:
                   carat        cut color clarity  ... price     x     y     z
           9       0.23  Very Good     H     VS1  ...   NaN     4.00  4.05  2.39
           89      0.32    Premium     I     SI1  ...   NaN     4.35  4.33  2.73
           1534    0.77      Ideal     E     SI1  ...   NaN     5.87  5.91  3.68
           21927   1.51       Good     H     VS2  ...   NaN     7.25  7.19  4.62
"""

# segregating the prediction data seperatly to a cut oriented dataframe called dfcut

dfcut = df[df['cut']=='Unknown']
dfcut.head()

"""
Output:
       carat      cut color clarity popularity  ...  table   price     x     y     z
53666   0.81  Unknown     J     VS2       Poor  ...   56.0  2708.0  5.92  5.97  3.69
53789   0.73  Unknown     D     SI2       Good  ...   56.0  2729.0  5.77  5.82  3.58
53925   0.79  Unknown     I     SI1       Good  ...   56.0  2756.0  5.95  5.97  3.67
"""

## We have succesfully segregated the prediction data, we will drop the NANs without any loss of required data.
## records of the dataset, we can drop them from the dataset itself.

df=df.dropna()
df.drop(df[df['cut'] =='Unknown'].index, inplace = True)

# Revaldiating after dropping the noisy data.
# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum())

print('\n*** Categorical Columns With Nulls ***')
print(df[categorical_features].isnull().sum()) 

print('\n*** Numerical Columns With Nulls ***')
print(df[numerical_features].isnull().sum())

print(df[df['cut']=='Unknown'])

"""
Ouput:
    
*** Columns With Nulls ***
carat         0
cut           0
color         0
clarity       0
popularity    0
depth         0
table         0
price         0
x             0
y             0
z             0
dtype: int64

*** Categorical Columns With Nulls ***
color         0
clarity       0
popularity    0
dtype: int64

*** Numerical Columns With Nulls ***
carat    0
depth    0
table    0
x        0
y        0
z        0
dtype: int64

print(df[df['cut']=='Unknown'])
Empty DataFrame
Columns: [carat, cut, color, clarity, popularity, depth, table, price, x, y, z]
Index: []

Observation?Reasoning:
    1. All the converted Null values are dropped from the dataset.
    2. All the invalid entries in the categorial features is cleaned from the dataset.

"""

## revalidating our prediction dataframes.
df.shape
dfprice.head()
dfcut.head()

"""
dfprice : Dataframe contains records to predict "price" as per section 4 of the project.
dfcut : Dataframe contains records to predict "cut" as per section 5 of the project.
"""

###############################################################################
# Handling Outliers.
###############################################################################

## Note : We cant run outlier handling before as any operation with NaN values are considered NaNs.

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df[numerical_features]))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df[numerical_features]))

"""
Output: (Only Outlier Count - Total 350 )
    *** Outlier Count ***
carat             40
depth             278
table             28
x                 0
y                 2
z                 2

Observation/Reasoning:
    1. Total of 350 outliers
    2. Earlier, we dropped 55 records(zeros and invalid entries.)
    3. We add, this figures w.r.t to records. total is 392 records.
    4. 392 is less than 1% of total records.
    5. From the information, we are given formula for depth and table. we can try calculating manually.
    6. We can also fill this with lower and upper limits.
    7. As, total records are very less compared to orignal dataset size. we dropped it.(less than 1%)
"""    


#handle outliers in depth and table.
#for column in numerical_features:
#    colValues = df[column].values
#    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
#    iqr = quartile_3 - quartile_1
#    lol, uol = (quartile_1 - (iqr * 3.0),quartile_3 + (iqr * 3.0))
#    print("Lower Outlier Limit:",lol)
#    print("Upper Outlier Limit:",uol)
#    df[column] = np.where(df[column] < lol, lol, df[column])
#    df[column] = np.where(df[column] > uol, uol, df[column])

# Alternate version to convert to NAN.
for column in numerical_features:
    print(column)
    colValues = df[column].values
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lol, uol = (quartile_1 - (iqr * 3.0),quartile_3 + (iqr * 3.0))
    print("Lower Outlier Limit:",lol)
    print("Upper Outlier Limit:",uol)
    df[column] = np.where(df[column] < lol, np.nan, df[column])
    df[column] = np.where(df[column] > uol, np.nan, df[column])

"""

Output:
    
    carat
Lower Outlier Limit: -1.52
Upper Outlier Limit: 2.96

    depth
Lower Outlier Limit: 56.5
Upper Outlier Limit: 67.0

    table
Lower Outlier Limit: 47.0
Upper Outlier Limit: 68.0

    x
Lower Outlier Limit: -0.7800000000000002
Upper Outlier Limit: 12.030000000000001

    y
Lower Outlier Limit: -0.7400000000000011
Upper Outlier Limit: 12.0

    z
Lower Outlier Limit: -0.47999999999999954
Upper Outlier Limit: 7.43

"""

### Validating the outliers in our data.

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df[numerical_features]))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df[numerical_features]))

"""
Output:
    
*** Outlier Count ***
carat             0
depth             0
table             0
x                 0
y                 0
z                 0


*** Outlier Values ***
carat 
[] 
depth 
[] 
table 
[] 
x 
[] 
y 
[] 
z 
[] 

"""


#Dropping this Null values.
df=df.dropna()

# Revalidating data
df.shape
# (53548, 11)

#### 1.b Show the structure and basic summary

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())
print(df.shape)

# summary
print("\n*** Summary ***")
print(df.describe())

"""
Observation:
# (53548, 11) new structure
# Orignal 53940, records were dropped to 53548 records after data cleaning. -- 392 records
# 7 records are prediction records moved to different datasets dfcut and dfprice. -- 385 records
# 385 records is less than 1% of total records.
# ***Important*** 420 values were either null or converted to NaN as part of data cleaning. (Asked in question).
"""
#visualdf = df.copy() ## Creating a copy to use for visualization later

########################################################################################
## Data transformation
########################################################################################
"""
## column 'cut' is allocated with color codes, we will proceed in grouping 
## and later coverting it into numerical feature.

#code_list = {
#        'D' : 'Colourless', 
#        'E' : 'Colourless', 
#        'F' : 'Colourless', 
#        'G' : 'Near_Colourless',
#        'H' : 'Near_Colourless',
#        'I' : 'Near_Colourless',
#        'J' : 'Near_Colourless',
#        'K' : 'Faint_Yellow',
#        'L' : 'Faint_Yellow',
#        'M' : 'Faint_Yellow',
#        'N' : 'Very_Light_Yellow',
#        'O' : 'Very_Light_Yellow',
#        'P' : 'Very_Light_Yellow',
#        'Q' : 'Very_Light_Yellow',
#        'R' : 'Very_Light_Yellow',
#        'S' : 'Light_Yellow',
#        'T' : 'Light_Yellow',
#        'U' : 'Light_Yellow',
#        'V' : 'Light_Yellow',
#        'W' : 'Light_Yellow',
#        'X' : 'Light_Yellow',
#        'Y' : 'Light_Yellow',
#        'Z' : 'Light_Yellow'}
#
#df['color'] = [code_list[x] for x in df['color']]

# Above commented, 
# As correlation is better without sub categorizing the data with dependent features rather than binning them.

# print(df.corr())

## 0.000236 & 0.172511
## -0.002966 & 0.144292
"""

## Handling other categorical variables.
## Placeholders, if required later to reverse map the codes to labels.
codeDicts = {}
codeDictsrev = {}

# get unique Class names
print("\n*** Unique Class - Categoric ***")
for column in categorical_features:
     print("Conversion for ", column)
     lnLabels = df[column].unique()
     print(lnLabels)
     df[column] = pd.Categorical(df[column])
     df[column] = df[column].cat.codes
     lnCCodes= df[column].unique()
     codeDicts.update({column:dict(zip(lnCCodes,lnLabels))})
     codeDictsrev.update({column:dict(zip(lnLabels,lnCCodes))})
print(codeDicts)

## Handling 'cut' column
    
lnLabels = df['cut'].unique()
print(lnLabels)
df['cut'] = pd.Categorical(df['cut'])
df['cut'] = df['cut'].cat.codes
lnCCodes= df['cut'].unique()
print(lnCCodes)
codeDicts.update({'cut':dict(zip(lnCCodes,lnLabels))})
codeDictsrev.update({'cut':dict(zip(lnLabels,lnCCodes))})
print(codeDicts['cut'])
#df['Labels'] = df['Species'].map(codeDicts)

print(codeDicts)
print(codeDictsrev)

"""
Output:
    {
    'color': {1: 'E', 5: 'I', 6: 'J', 4: 'H', 2: 'F', 3: 'G', 0: 'D'}, 
    'clarity': {3: 'SI2', 2: 'SI1', 4: 'VS1', 5: 'VS2', 7: 'VVS2', 6: 'VVS1', 0: 'I1', 1: 'IF'}, 
    'popularity': {1: 'Good', 0: 'Fair', 2: 'Poor'}, 
    'cut': {2: 'Ideal', 3: 'Premium', 1: 'Good', 4: 'Very Good', 0: 'Fair'}
    }
    {
    'color': {'E': 1, 'I': 5, 'J': 6, 'H': 4, 'F': 2, 'G': 3, 'D': 0}, 
    'clarity': {'SI2': 3, 'SI1': 2, 'VS1': 4, 'VS2': 5, 'VVS2': 7, 'VVS1': 6, 'I1': 0, 'IF': 1}, 
    'popularity': {'Good': 1, 'Fair': 0, 'Poor': 2}, 
    'cut': {'Ideal': 2, 'Premium': 3, 'Good': 1, 'Very Good': 4, 'Fair': 0}
    }
    
Observation/Reasoning:
1. To reverse map the numberic codes to appropriate labels.    
"""

########################################################################################
## Section 3
## Visual Data Analysis
########################################################################################
"""
## Data distribution.
## WE will plot histograms for the continous features.
## WE will use Plot bars for the discreate features.
"""

continous_features = ['price','carat']
discrete_features = ['color', 'cut', 'clarity']

# histograms
# plot histograms
print("\n*** Histogram Plot ***")
print('Histograms')
for colName in continous_features:
    colValues = df[colName].values
    plt.figure(figsize=(15,8))
    sns.distplot(colValues, bins=9, kde=False)
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

"""
## OBSERVATIONS
# The displot gives the distribution of price & Carat
# Most of diamonds are having their price between 1000-2500, higher price diamonds are lesser in count.
# Most of diamonds are having carats between 0.2 - 1.0 carat value. 
# Potential relationship is Carat and Price is directly proportional to eachother.
"""

# class count plot
print("\n*** Distribution Plot ***")
for colName in discrete_features:
    plt.figure(figsize=(15,8))
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Count')
    plt.show()

"""
## OBSERVATIONS
# The countplot gives the data distribution of discrete features 'color', 'cut' & 'clarity'
# print(codeDicts['color'])
# {1: 'E', 5: 'I', 6: 'J', 4: 'H', 2: 'F', 3: 'G', 0: 'D'} -- Color

#####################

# The color is quite evenly distributed, color type 'G' has the most entries in the dataset.
# print(codeDicts['cut'])
# {2: 'Ideal', 3: 'Premium', 1: 'Good', 4: 'Very Good', 0: 'Fair'}-- cut
# Most of the diamonds have cut type as 'Ideal'

#####################

# print(codeDicts['clarity'])
# {3: 'SI2', 2: 'SI1', 4: 'VS1', 5: 'VS2', 7: 'VVS2', 6: 'VVS1', 0: 'I1', 1: 'IF'} -- clarity
# SI1 & SI2 have the most entried while I1 is very low in count.
"""

for c in ['color', 'cut', 'clarity']:
    data=df.copy()
    plt.figure(figsize=(16,8))
    sns.barplot(x=c, y="price", data=df)
    plt.xticks(rotation=-45)
    if c == 'color':
        plt.legend(bbox_to_anchor=(1.05, 1), loc ='upper left', borderaxespad=0,title='Color type',
                   prop={'size': 'x-large'},
                   labels=['1:E', '5:I', '6:J', '4:H', '2:F', '3:G', '0:D'])
    elif c == 'cut':
        plt.legend(bbox_to_anchor=(1.05, 1),loc ='upper left', borderaxespad=0,title='Cut type', 
                   prop={'size': 'x-large'},
                   labels=['0:Fair', '1:Good','2:Ideal', '3:Premium', '4:Very Good'])
    elif c == 'clarity':
        plt.legend(bbox_to_anchor=(1.05, 1), loc ='upper left', borderaxespad=0,title='Clarity type',
                   prop={'size': 'x-large'},
                   labels=
                   [ '0: I1', '1: IF', '2: SI1','3: SI2', '4: VS1', '5: VS2','6: VVS1', '7: VVS2'])
    plt.xlabel(c)
    plt.ylabel('price')
    title = "Relationship between Price and " + str(c).capitalize()
    plt.title(title)
    plt.show()
    
"""

## OBSERVATIONS
# The barplot for discrete numerical features with 'color', 'cut' & 'clarity'. Repesentation with central tendency
# w.r.t to mean or average.
    
    ######################
# The graph gives the price comparison for each color type
# With respect to color : Type J : Near Colorless category is higher in price than Type D: Colorless Class.
# Type E still seems to be better than Type D color.
    
    ######################
# The graph gives the price comparison for each cut type
# With respect to color : Premium is the most expensive ones
# The prices are quite even, Cut doesnt seem to hold a good weight in terms of price prediction and vice versa.
    
    ######################
# The graph gives the price comparison for each clarity type
# SI2 leads among other types and IF1 and VVS1 types are least expensive.

"""


"""
#To check the distibution of diamonds based on price,cut,carat
#price with carat gives a linear relation, which is directly proportional, 
# rest of the columns are not having any linear relation
"""

print("\n*** Scatter Plot ***")
dfn1=df
plt.figure(figsize=(12,12))
sns.pairplot(dfn1,size=2);
plt.show()

# check relation with corelation - table
pd.options.display.float_format = '{:,.2f}'.format
print(dfn1.corr())

# check relation with corelation - heatmap
plt.figure(figsize=(12,12))
sns.heatmap(dfn1.corr(), annot=True)
plt.show()

"""
## As noticed earlier there is strong positive correlation with Carat and Price.
## Carat is basically its mass as 200mg.
## Hence, we see a positive relation with x, y, and z columns which are lenght width and depth of the diamond.
## Higher the weight and size, the costier the diamond will be.
"""

plt.figure(figsize=(15,8))
sns.regplot(x="carat", y="price", data=dfn1, color='b')
plt.title("Carat v/s Price w/ Trendline")
plt.ylabel('Price')
plt.xlabel('Carat')
plt.show()

plt.figure(figsize=(15,8))
df.plot(kind='box',figsize=(15,10),subplots=True)
plt.show()


"""
"cut" is dependent on features : price, table & width.
Checking relationship.
"""


for feature in ['price', 'table','depth']:
    data=df.copy()
    data.groupby('cut')[feature].median().plot.bar()
    plt.xlabel('cut')
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()

"""
1. 'depth' and 'table' are pretty flat in terms of relationship with cut.
2. constant or no specific relation is observed.
3. Category code 2: Ideal are the cheapest in price and Fair are most expensive.
"""

########################################################################################
## Section 4
## Linear Regression
########################################################################################

"""
Reasoning:
    
1. It's okay for linear regression to not have normalised data, so we will skip normalization or standardization here.
2. We have created another dataframe called "cleanDf", 
   we will be using it as input for both section 4 and 5(LR and Classification).
   
"""
#Storing the clean data in seperate data frame.
cleanDf = df.copy() ## One Time

df = cleanDf.copy() ## Use only this.

################################
# Classification 
# Split Train & Test
###############################

# split into data & target
print("\n*** Prepare Data ***")
dftrain = df.sample(frac=0.8, random_state=707)
dftest=df.drop(dftrain.index)
print("Train Count:",len(dftrain.index))
print("Test Count :",len(dftest.index))

"""
Output:
    *** Prepare Data ***
    Train Count: 42838
    Test Count : 10710
"""

##############################################################
# Model Creation & Fitting And Prediction for Feature 
##############################################################

# all cols except 'price' as its the dependent variable.
print("\n*** Regression Data ***")
allCols = df.columns.tolist()
print(allCols)
#allCols.remove('carat')
allCols.remove('price')
#allCols.remove('depth')
#allCols.remove('table')
#allCols.remove('x')
#allCols.remove('y')
#allCols.remove('z')
print(allCols)

# regression summary for feature
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(dftrain[allCols])
y = dftrain['price']
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# Observation
# 1. R-squared value 0.896 is higher than 0.65.
# 2. p-value is higher than 0.05 for depth(0.787), we will drop it.

allCols.remove('depth')
print(allCols)

# regression summary for feature
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(dftrain[allCols])
y = dftrain['price']
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# Observation
# 1. R-squared value is 0.89 higher than 0.65, which means 89% variablility is explained by the model.
# 2. All p-value's are lower than 0.05, we will proceed further.

# now create linear regression model

print("\n*** Regression Model ***")
X = dftrain[allCols].values
y = dftrain['price'].values
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X,y)
print(model)

# predict
p = model.predict(X)
dftrain['predict'] = p

##############################################################
# Model Evaluation - Train Data
##############################################################

print("R2  : {}".format(np.sqrt(r2_score((y),(p)))))

## R2 score for training dataset with Linear Regression is 94%

# visualize 
print("\n*** Scatter Plot ***")
plt.figure(figsize=(12,12))
sns.regplot(data=dftrain, x='price', y='predict', color='b', scatter_kws={"s": 5})
plt.show()

# mae 
print("\n*** Mean Absolute Error ***")
mae = mean_absolute_error(dftrain['price'], dftrain['predict'])
print(mae)

## *** Mean Absolute Error ***
## 823.8802322731945

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(dftrain['price'], dftrain['predict'])
print(mse)

## *** Mean Squared Error ***
## 1655744.0272107255
   
# rmse 
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

## *** Root Mean Squared Error ***
## 1286.7571749210204

# check mean
print('\n*** Mean ***')
print(dftrain['price'].mean())
print(dftrain['predict'].mean())

## *** Mean ***
## 3919.9852700873057
## 3919.985270087277

# scatter index (SI) is defined to judge whether RMSE is good or not. 
# SI=RMSE/measured data mean. 
# If SI is less than one, your estimations are acceptable.
print('\n*** Scatter Index ***')
si = rmse/dftrain['price'].mean()
print(si)

"""
## Scatter Index value is 0.3282556148208082 is less than 1, It is good and our model is acceptable.
"""

##############################################################
# confirm with test data 
##############################################################

# all cols except price 
print("\n*** Regression Data For Test ***")
print(allCols)
# split
X = dftest[allCols].values
y = dftest['price'].values
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# predict
p = model.predict(X)
dftest['predict'] = p

##############################################################
# Model Evaluation - Test Data
##############################################################

print("R2  : {}".format(np.sqrt(r2_score((y),(p)))))

#### R2 score for test dataset with Linear Regression is 94%

# visualize 
print("\n*** Scatter Plot ***")
plt.figure(figsize=(12,12))
sns.regplot(data=dftest, x='price', y='predict', color='b', scatter_kws={"s": 5})
plt.show()

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(dftest['price'], dftest['predict'])
print(mae)

# *** Mean Absolute Error ***
# 822.944526494159

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(dftest['price'], dftest['predict'])
print(mse)
   
# *** Mean Squared Error ***
# 1647329.3604883647

# rmse 
# RMSE measures the error.  How good is an error depends on the amplitude of your data. 
# RMSE should be less 10% for mean(depVars)
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# *** Root Mean Squared Error ***
# 1283.4832918617853

# check mean
print('\n*** Mean ***')
print(dftrain['price'].mean())
print(dftrain['predict'].mean())
 
# scatter index
# scatter index less than 1; the predictions are decent
print('\n*** Scatter Index ***')
si = rmse/dftest['price'].mean()
print(si)

"""
## Scatter Index value is 0.32590405506630543 is less than 1, It is good and our model is acceptable.
"""

##############################################################
# Final Prediction
##############################################################

dffinal=dfprice.copy() ## Creating new dataframe out of dfprice where our final records for price prediction.

print(dffinal)

"""
Output: Before converting the categories to numbers.
       carat        cut color clarity popularity  ...  table  price    x    y    z
9       0.23  Very Good     H     VS1       Fair  ...  61.00    nan 4.00 4.05 2.39
89      0.32    Premium     I     SI1       Good  ...  58.00    nan 4.35 4.33 2.73
1534    0.77      Ideal     E     SI1       Good  ...  56.00    nan 5.87 5.91 3.68
21927   1.51       Good     H     VS2       Poor  ...  59.00    nan 7.25 7.19 4.62
"""

## Apply all the data cleaning and transformation as per the earlier train data set.

dffinal['cut'] = dffinal['cut'].map(codeDictsrev['cut'])
dffinal['color'] = dffinal['color'].map(codeDictsrev['color'])
dffinal['clarity'] = dffinal['clarity'].map(codeDictsrev['clarity'])
dffinal['popularity'] = dffinal['popularity'].map(codeDictsrev['popularity'])

print(dffinal)

"""
Output: After converting the categories to numbers.
       carat  cut  color  clarity  popularity  ...  table  price    x    y    z
9       0.23    4      4        4           0  ...  61.00    nan 4.00 4.05 2.39
89      0.32    3      5        2           1  ...  58.00    nan 4.35 4.33 2.73
1534    0.77    2      1        2           1  ...  56.00    nan 5.87 5.91 3.68
21927   1.51    1      4        5           2  ...  59.00    nan 7.25 7.19 4.62

"""

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(dffinal[numerical_features]))


# all cols except price column
print("\n*** Regression Data For Predicted values ***")
allCols = dffinal.columns.tolist()
print(allCols)
allCols.remove('price')
allCols.remove('depth')
#allCols.remove('table')
#allCols.remove('x')
#allCols.remove('y')
#allCols.remove('z')
print(allCols)

"""
Observations:
1 . Pre-requisties, like zeros, nan in model data, outliers or anything to fail are model is not present.
2 . We are ready to predict.
"""

# predict
X = dffinal[allCols].values
p = model.predict(X)

dffinal=dfprice.copy()
dffinal['price'] = p

print(dffinal)

"""
Output:
#        carat        cut color clarity  ...        price     x     y     z
# 9       0.23  Very Good     H     VS1  ...  -181.644608  4.00  4.05  2.39
# 89      0.32    Premium     I     SI1  ... -1031.118814  4.35  4.33  2.73
# 1534    0.77      Ideal     E     SI1  ...  3316.593378  5.87  5.91  3.68
# 21927   1.51       Good     H     VS2  ...  9317.175676  7.25  7.19  4.62

# Observations:

# If we perform further analysis to increase the accuracy, we can drop table column to increase the accuracy slightly and
# get a positive price Index 9 as $85.

#       carat        cut color clarity  ...        price     x     y     z
# 9       0.23  Very Good     H     VS1  ...    85.910848  4.00  4.05  2.39
# 89      0.32    Premium     I     SI1  ...  -954.669540  4.35  4.33  2.73
# 1534    0.77      Ideal     E     SI1  ...  3239.483078  5.87  5.91  3.68
# 21927   1.51       Good     H     VS2  ...  9516.888615  7.25  7.19  4.62
    
"""

########################################################################################
## Section 5.
## Classification Model to predict cut.
########################################################################################

# Load the clean data
df = cleanDf.copy() ## Use only this.

# cut is dependent on price,depth and Table features

clsVars = 'cut' ## Set the prediction feature.
cols = ['price','depth','table']
#cols = ['price','depth','table','color','clarity']
#cols = df.columns.tolist()
#cols.remove(clsVars)

## Feature engineering to bring it down to percentage scale.
# df['price'] = (df['price'].values*100)/df['price'].max()

#53939
df1=df.copy()

"""
1. It is important to Standardize/Normalize the data, Normalization is required
2. For the euclidian distant based or alogorithms that uses gradient descent parabola.
"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# fit
ar = scaler.fit_transform(df[cols])

# transform generates an np array ... so back to df
df = df.reset_index() ## Its important as sometimes you will see NAN values after normalization/standardization
df[cols] = pd.DataFrame(data=ar)

print(df.isnull().sum())
"""
index         0
carat         0
cut           0
color         0
clarity       0
popularity    0
depth         0
table         0
price         0
x             0
y             0
z             0
dtype: int64
"""

for col in cols:
    fig, ax=plt.subplots(1,2)
    sns.distplot(df1[col], ax=ax[0])
    ax[0].set_title("Original Data")
    sns.distplot(df[col], ax=ax[1])
    ax[1].set_title("Scaled data")

################################
# Classification 
# set X & y
###############################

# split into data & target
print("\n*** Prepare Data ***")
X = df[cols].values
y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])


print(codeDicts)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())
print((df.groupby(clsVars).size()/df[clsVars].count())*100)

print('Percentage wise : {0}'.format(np.round(df.groupby(clsVars).size()/df[clsVars].count()*100,2)))

"""
Output:
    
    Percentage wise : cut
    0    2.46
    1    9.11
    2   40.21
    3   25.68
    4   22.54
    dtype: float64
    
"""

## Oversampling to get the multi label classification balanced.
## Either one item.

# import
from imblearn.over_sampling import RandomOverSampler
# create os object
os =  RandomOverSampler(random_state = 707)
# generate over sampled X, y
X,y = os.fit_sample(X, y)


# import
# from imblearn.over_sampling import SMOTE
# create smote object -- Gives 71% accuracy for random forest.
# sm = SMOTE(random_state = 707)
# generate over sampled X, y
# X,y = sm.fit_resample(X, y)

# counts
print("\n*** Counts ***")
unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# *** Counts ***
# [[    0     1     2     3     4]
# [21531 21531 21531 21531 21531]]


# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

"""
Output:
    
*** Prepare Data - Shape ***
(107655, 3)
(107655,)
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
"""

################################
# Classification 
# Split Train & Test
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.33, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))

"""
Output:
    
    *** Frequency of unique values of Train Data ***
    [[    0     1     2     3     4]
    [14426 14425 14426 14425 14426]]

    *** Frequency of unique values of Test Data ***
    [[   0    1    2    3    4]
    [7105 7106 7105 7106 7105]]
"""


################################
# Classification 
# actual model ... create ... fit ... predict
###############################

# original
# import all model & metrics
print("\n*** Importing Models ***")
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
print("\nDone ...")

# create a list of models so that we can use the models in an iterstive manner
print("\n*** Creating Models ***")
models = []
models.append(('SVM-Clf', SVC()))
models.append(('RndFrst', RandomForestClassifier(random_state=707)))
models.append(('KNN-Clf', KNeighborsClassifier()))
models.append(('LogRegr', LogisticRegression()))
models.append(('DecTree', DecisionTreeClassifier()))
models.append(('GNBayes', GaussianNB()))
print(models)
print("\nDone ...")


################################
# Classification 
# Cross Validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvAccuracy = []
xvSDScores = []
print("Done ...")


# cross validation
from sklearn import model_selection
print("\n*** Cross Validation ***")
# iterate through the models
for name, model in models:
    # select xv folds
    kfold = model_selection.KFold(n_splits=10, random_state=707)
    # actual corss validation
    cvAccuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    #cvAccuracy = cross_val_score(RandomForestClassifier(random_state=707), X_train.values(), y_train.values(), cv=kfold, scoring='accuracy')
    # prints result of cross val ... scores count = lfold splits
    print(name,":  ",cvAccuracy)
    # update lists for future use
    xvModNames.append(name)
    xvAccuracy.append(cvAccuracy.mean())
    xvSDScores.append(cvAccuracy.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%7s: %10s %8s" % ("Model", "xvAccuracy", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(xvModNames)):
    # print accuracy mean & std
    msg = "%8s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV Accuracy Model ***")
maxXVIndx = xvAccuracy.index(max(xvAccuracy))
print("Index     ",maxXVIndx)
print("Model Name",xvModNames[maxXVIndx])
print("XVAccuracy",xvAccuracy[maxXVIndx])
print("XVStdDev  ",xvSDScores[maxXVIndx])
print("Model     ")
print(models[maxXVIndx])

"""
Output:
## Using RandomOverSampler 
## *** Cross Validation Summary ***
##  Model: xvAccuracy xvStdDev
## SVM-Clf: 0.7031945 0.0061269
## RndFrst: 0.8532469 0.0036786
## KNN-Clf: 0.7530640 0.0056250
## LogRegr: 0.4785658 0.0042849
## DecTree: 0.8402840 0.0039559
## GNBayes: 0.6319461 0.0049883
##
## *** Best XV Accuracy Model ***
## Index      1
## Model Name RndFrst
## XVAccuracy 0.8532469164452333
## XVStdDev   0.003678599677700631
## Model     
## ('RndFrst', RandomForestClassifier(random_state=707))
"""

################################
# Classification 
# evaluate : Accuracy & Confusion Metrics
###############################

# print original confusion matrix
print("\n*** Confusion Matrix ***")
cm = confusion_matrix(y_test, y_test)
print("Original")
print(cm)

# blank list to hold info
print("\n*** Confusion Matrix - Init ***")
cmModelInf = []
cmModNames = []
cmAccuracy = []
print("\nDone ... ")

# iterate through the modes and calculate accuracy & confusion matrix for each
print("\n*** Confusion Matrix - Compare ***")
for name, model in models:
    # fit the model with train dataset
    model.fit(X_train, y_train)
    # predicting the Test set results
    y_pred = model.predict(X_test)
    # accuracy
    Accuracy = accuracy_score(y_test, y_pred)
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # X-axis Predicted | Y-axis Actual
    print("")
    print(name)
    print(cm)
    print("Accuracy", Accuracy)
    # update lists for future use 
    cmModelInf.append((name, model, cmAccuracy))
    cmModNames.append(name)
    cmAccuracy.append(Accuracy)

# cross val summary
print("\n*** Confusion Matrix Summary ***")
# header
msg = "%7s: %10s " % ("Model", "xvAccuracy")
print(msg)
# for each model
for i in range(0,len(cmModNames)):
    # print accuracy mean & std
    msg = "%8s: %5.7f" % (cmModNames[i], cmAccuracy[i])
    print(msg)

print("\n*** Best CM Accuracy Model ***")
maxCMIndx = cmAccuracy.index(max(cmAccuracy))
print("Index     ",maxCMIndx)
print("Model Name",cmModNames[maxCMIndx])
print("CMAccuracy",cmAccuracy[maxCMIndx])
print("Model     ")
print(models[maxCMIndx])
blnGridSearch = False

"""
Output:
    
    Confusion matrix for Random Forest.
    RndFrst
[[7105    0    0    0    0]
 [   7 7018    0    0   81]
 [   4   66 5679  572  784]
 [  12  119  407 5735  833]
 [  14  346  668  892 5185]]

Accuracy 0.8647507529484617

*** Confusion Matrix Summary ***
  Model: xvAccuracy 
 SVM-Clf: 0.7037183
 RndFrst: 0.8647508
 KNN-Clf: 0.7599009
 LogRegr: 0.4803107
 DecTree: 0.8555746
 GNBayes: 0.6346722

*** Best CM Accuracy Model ***
Index      1
Model Name RndFrst
CMAccuracy 0.8647507529484617
Model     
('RndFrst', RandomForestClassifier(random_state=707))
    
"""

################################
#Finding best parameters for our model
###############################

from sklearn.model_selection import GridSearchCV
print("\n*** Grid Search XV For " + cmModNames[maxCMIndx] + " ***")
blnGridSearch = True
# SVC model
if cmModNames[maxCMIndx] == 'SVM-Clf':
    param = { 
        'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
        'kernel':['linear', 'rbf'],
        'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
    }
    gsxv = GridSearchCV(SVC(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# random forest model
if cmModNames[maxCMIndx] == 'RndFrst':
    param = { 
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth'   : [4,5,6,7,8],
        'criterion'   : ['gini', 'entropy']
    }
    gsxv = GridSearchCV(RandomForestClassifier(random_state=707), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# knn classifir
if cmModNames[maxCMIndx] == 'KNN-Clf':
    param = { 
        'leaf_size' : list(range(1,50)), 
        'n_neighbors' : list(range(1,30)), 
        'p' : [1,2]
    }
    gsxv = GridSearchCV(KNeighborsClassifier(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# logistic regression
if cmModNames[maxCMIndx] == 'LogRegr':
    param = { 
        'C':[0.001,.009,0.01,.09,1,5,10,15,20,25],
        'penalty': ['l1', 'l2',13,14,15]        
    }
    gsxv = GridSearchCV(LogisticRegression(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# decision tree
if cmModNames[maxCMIndx] == 'DecTree':
    param = { 
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth'   : [4,5,6,7,8],
        'criterion'   : ['gini', 'entropy']
    }
    gsxv = GridSearchCV(DecisionTreeClassifier(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# naive bayes
if cmModNames[maxCMIndx] == 'GNBayes':
    param = { 
        'vec__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),  
        'vec__max_features': (None, 5000, 10000, 20000),  
        'vec__min_df': (1, 5, 10, 20, 50),  
    }        
    gsxv = GridSearchCV(GaussianNB(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
# show verbose    
gsxv.fit(X_train, y_train)
bestParams = gsxv.best_params_
print(bestParams)
print(type(bestParams))

"""
Output:
    [Parallel(n_jobs=1)]: Done 300 out of 300 | elapsed: 16.1min finished
    {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 100}
    <class 'dict'>
"""

################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", models[maxXVIndx]) 
print("Confusion Matrix")
print("Accuracy:", cmAccuracy[maxCMIndx])
print("Model   :", models[maxCMIndx]) 

"""
Output:
    *** Accuracy & Models ***
    Cross Validation
    Accuracy: 0.8533439576767128
    Model   : ('RndFrst', RandomForestClassifier(random_state=707))
    Confusion Matrix
    Accuracy: 0.8647507529484617
    Model   : ('RndFrst', RandomForestClassifier(random_state=707))
"""

# classifier object
print("\n*** Classfier Object ***")
if ~blnGridSearch:
    cf = models[maxCMIndx][1]
else:
    if cmModNames[maxCMIndx] == 'SVM-Clf':
        cf = SVC(C=bestParams['C'], kernel=bestParams['kernal'], gamma=bestParams['gamma'])
    if cmModNames[maxCMIndx] == 'RndFrst':
        cf = RandomForestClassifier(criterion=bestParams['criterion'], max_depth=bestParams['max_depth'], max_features=bestParams['max_features'], n_estimators=bestParams['n_estimators'], random_state=707)
    if cmModNames[maxCMIndx] == 'KNN-Clf':
        cf = KNeighborsClassifier(leaf_size=bestParams['leaf_size'], n_neighbors=bestParams['n_neighbors'], p=bestParams['p'])
    if cmModNames[maxCMIndx] == 'LogRegr':
        cf = LogisticRegression(C=bestParams['C'], penalty=bestParams['penalty'])
    if cmModNames[maxCMIndx] == 'DecTree':
        cf = DecisionTreeClassifier(max_features=bestParams['max_features'],max_depth=bestParams['max_depth'],criterion=bestParams['criterion'])
    if cmModNames[maxCMIndx] == 'GNBayes':
        cf = GaussianNB(vec__max_df=bestParams['vec__max_df'],vec__max_features=bestParams['vec__max_features'],vec__min_df=bestParams['vec__min_df'])
print(cf)
# fit the model
cf.fit(X_train,y_train)
print("Done ...")

# classifier object
print("\n*** Predict Test ***")
# predicting the Test set results
y_pred = cf.predict(X_test)            # use model ... predict
print("Done ...")

# accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_test, y_pred)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test,y_pred)
print(cr)


"""
Output:
    
    *** Classfier Object ***
    RandomForestClassifier(random_state=707)
    Done ...

    *** Predict Test ***
    Done ...

    *** Accuracy ***
    86.47507529484616

    *** Confusion Matrix - Original ***
    [[7105    0    0    0    0]
    [   0 7106    0    0    0]
    [   0    0 7105    0    0]
    [   0    0    0 7106    0]
    [   0    0    0    0 7105]]

    *** Confusion Matrix - Predicted ***
    [[7105    0    0    0    0]
    [   7 7018    0    0   81]
    [   4   66 5679  572  784]
    [  12  119  407 5735  833]
    [  14  346  668  892 5185]]

    *** Classification Report ***
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      7105
           1       0.93      0.99      0.96      7106
           2       0.84      0.80      0.82      7105
           3       0.80      0.81      0.80      7106
           4       0.75      0.73      0.74      7105

     accuracy                           0.86     35527
    macro avg       0.86      0.86      0.86     35527
 weighted avg       0.86      0.86      0.86     35527
"""

################################
# Final Prediction
###############################

dffinal=dfcut.copy() ## Creating new dataframe out of dfcut where our final records for "cut" prediction.

print(dffinal)

## Apply all the data cleaning and transformation as per the earlier train data set.
#df['Labels'] = df['Species'].map(codeDicts)
#print(codeDicts)
#dffinal['color'] = dffinal['color'].map(codeDictsrev['color'])
#dffinal['clarity'] = dffinal['clarity'].map(codeDictsrev['clarity'])
#dffinal['popularity'] = dffinal['popularity'].map(codeDictsrev['popularity'])
#print(dffinal)

print(cols) ## ['price', 'depth', 'table']

# predict ## Run the model on the main dataframe
X_pred = dffinal[cols].values

# y_pred holds the predicted values of the random forest classification model.
y_pred = cf.predict(X_pred)  


dffinal=dfcut.copy()
dffinal['cut'] = y_pred

# reverse mapping the values
dffinal['cut'] = dffinal['cut'].map(codeDicts['cut'])

print(dffinal)

## All the three "unknown" records are predicted as "FAIR"

##       carat    cut color clarity popularity  ...  table    price    x    y    z
##53666   0.81  Fair     J     VS2       Poor  ...  56.00 2,708.00 5.92 5.97 3.69
##53789   0.73  Fair     D     SI2       Good  ...  56.00 2,729.00 5.77 5.82 3.58
##53925   0.79  Fair     I     SI1       Good  ...  56.00 2,756.00 5.95 5.97 3.67


#########################################################################################################
## Classification Model to predict cut.(PCA) - Part of trying to increase accuracy. Not applicable though
#########################################################################################################

df = cleanDf.copy() ## Use only this.
clsVars = 'cut' ## Set the prediction feature.

# cut is dependent on price,depth and Table features
cols = ['price','depth','table']

X = df[cols].values
y = df[clsVars].values

# Feature extraction with selectBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# feature extraction
model = SelectKBest(score_func=f_classif, k=3)
fit = model.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
# data frame
dfm =  pd.DataFrame({'Cols':cols, 'Imp':fit.scores_})  
dfm.sort_values(by='Imp', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
print(dfm)

# feature extraction with ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
# extraction
model = ExtraTreesClassifier(n_estimators=10, random_state=707)
model.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(model.feature_importances_)
# data frame
dfm =  pd.DataFrame({'Cols':cols, 'Imp':model.feature_importances_})  
dfm.sort_values(by='Imp', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
print(dfm)

#    Cols       Imp
# 0  price  0.434330
# 1  depth  0.289415
# 2  table  0.276255

"""
### ~~~~~~~~~~~~~~~ Final Observation, Resoning and Interpretation Summary ~~~~~~~~~~~~~~
## Section 5 : ~~~
		1. 85% accuracy : random forest classifciation algorigthm after applying the over sampling(random sampler).
		Hence, Random Forest is best classification model for prediction.
		2. Based on Price, Depth and Table.
        
        ~~~ After trying different combination and guestimates to improve the accuracy. ~~~
        
		3. 66% accuracy : random forest classifciation algorigthm with no modifications for train and test.
		4. 66% accuracy : random forest classifciation algorigthm with no modifications for train and test(CV Search Grid).
		5. 67% KNeighborsClassifier os best model after converting price column on a scale of 100.(test and train).
		6. 71% accuracy : random forest classifciation algorigthm after applying the over sampling(SMOTE).
		7. Normalization improves accuracy for KNN, SVC and logistic regression in our model. Due to gradient descent.
		8. With all the data, the classification model acheives 77% accuracy.
        
## Section 4 : ~~~
## 1. 95% accuracy : Linear Regression algorigthm.
## 2. We can increase the accuracy further by dropping less correlated feature like table.

## Section 3 : ~~~
    1. Data distribution:
        a. Continous features - Histogram.
        b. Discrete features - Barplot.
    2. Price relationship with barplots with price on y axis and other C's (discrete features) on X-axis.
    3. Scatter plot to identify any unique relationships.
        a. Regplot with trendline between carat and price. strong correlation.
    4. VDA with respect to "CUT" and Dependent features - price, table, depth.

## Section 2 : ~~~
        1. Data with zeros converted to NaN.
        2. Invalid categorical data converted to NaN.
        3. Outliers converted to NaN, and NaN values dropped. -- 420 total NaN values (either present + converted to NaNs)
        4. Orignal 53940 records, were reduced to 53548 records after data cleaning. -- 392 records
        5. 392 records are less than 1% of total dataset.
        
## Section 1 : ~~~
        1. 53940 total records after reading, with 12 columns.
"""
















































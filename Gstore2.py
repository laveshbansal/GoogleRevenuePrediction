import os
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')



def load_df(csv_path='C:/Users/Lavesh/.PyCharm2018.2/config/scratches/GStore_Data/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    p=0.1
    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'}, nrows=nrows, # Important!!
                      skiprows=lambda i: i > 0 and random.random() > p)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

train_df = load_df()
pd.set_option('display.max_columns', None)
# print(train_df.head())
# shops_or_not=lambda x : x.train_df.totals.transactionRevenue > 0
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
train_df['shops or not'] = train_df['totals.transactionRevenue'].values > 0
# y_clf = (train_df['totals.transactionRevenue'].fillna(0) > 0).astype(np.uint8)
# print(pd.value_counts(train_df['shops or not']))
# print(pd.value_counts(y_clf))

def date_format(df):
    df['date'] = pd.to_datetime(df['date'])
    df['vis_date'] = pd.to_datetime(df['visitStartTime'])
    df['sess_date_dow'] = df['vis_date'].dt.dayofweek
    df['sess_date_hours'] = df['vis_date'].dt.hour
    df['sess_date_dom'] = df['vis_date'].dt.day

date_format(train_df)

print(train_df.shape)

excluded_features =['socialEngagementType','totals.visits','totals.newVisits','device.browserVersion','device.flashVersion',
         'device.language','device.browserSize','device.mobileDeviceInfo','device.mobileDeviceMarketingName',
         'device.mobileDeviceModel','device.mobileInputSelector','device.operatingSystemVersion',
         'device.screenColors','device.screenResolution','device.mobileDeviceBranding','geoNetwork.cityId',
          'geoNetwork.latitude','geoNetwork.longitude','geoNetwork.networkLocation',
          'trafficSource.adwordsClickInfo.criteriaParameters',
                    # 'trafficSource.campaignCode',
            'trafficSource.campaign','trafficSource.adContent','trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adwordsClickInfo.gclId','trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.adwordsClickInfo.page','trafficSource.adwordsClickInfo.slot']

A=train_df.drop(excluded_features,axis=1)
print(A.shape)

import matplotlib.pyplot as plt
import seaborn as sns
# devicecategory
pd.value_counts(A['device.deviceCategory'])
pl=sns.countplot(A['device.deviceCategory'])
pl.set_title('Device Usage')
plt.show()

# Browser
pd.value_counts(A['device.browser'])
plt.figure(figsize=(10,6))
pl = sns.countplot(x='device.browser',data=A[A['device.browser'].isin(A['device.browser'].value_counts()[:7].index)])
pl.set_title('Browser Usage')
pl.set_xlabel('Browser Name')
pl.set_ylabel('Count')
plt.show()

#Subcontinent
plt.figure(figsize=(17,6))
pl = sns.countplot(x = 'geoNetwork.subContinent' , data = A[A['geoNetwork.subContinent'].isin(A['geoNetwork.subContinent'].value_counts()[:15].index)],palette="hls")
pl.set_title("Top 15 most frequent Sub Continents")
pl.set_xlabel("Sub Continent")
pl.set_ylabel("Count")
plt.xticks(rotation=45)
plt.show()

#OS
plt.figure(figsize=(10,6))
pl = sns.countplot(x='device.operatingSystem',data = A[A['device.operatingSystem'].isin(A['device.operatingSystem'].value_counts()[:8].index)])
pl.set_title("Usage of Operating System")
pl.set_xlabel("Operating System")
pl.set_ylabel("Count")
plt.show()

#Revenue vs Browser
pl = (A[A['totals.transactionRevenue'] > 0].groupby(['device.browser'])[['totals.transactionRevenue']]
          .sum().sort_values(by='totals.transactionRevenue',ascending=False)[:7]).plot.bar()
pl.set_title("Total Revenue v/s Browser")
pl.set_xlabel("Browser")
pl.set_ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()

# Revenue vs OS
pl = (A[A['totals.transactionRevenue'] > 0].groupby(['device.operatingSystem'])[['totals.transactionRevenue']]
          .sum().sort_values(by='totals.transactionRevenue',ascending=False)[:8]).plot.bar()
pl.set_title("Total Revenue v/s Operating System")
pl.set_xlabel("Operating System")
pl.set_ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()


#Source - NLP
from wordcloud import WordCloud
plt.figure(figsize=(10,7))
wordcloud = WordCloud(
                          max_words=30,
                          max_font_size=45
                         ).generate(' '.join(A['trafficSource.source']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Most frequent used Traffic Source")
plt.show()

# Browsers used in OS
crossplot = pd.crosstab(index=A[A['device.operatingSystem'].isin(A['device.operatingSystem'].value_counts()[:6].index.values)]['device.operatingSystem'],
                          columns=A[A['device.browser'].isin(A['device.browser'].value_counts()[:5].index.values)]['device.browser'])
crossplot.plot(figsize=(10,10),kind='bar',stacked=True)
plt.title("Most frequent OS's by Browsers of users")
plt.xlabel("Operating System")
plt.ylabel("Count OS")
plt.xticks(rotation=0)
plt.show()


replace_null_values={'trafficSource.isTrueDirect': 'False', 'trafficSource.keyword': 'unknown',
                     'trafficSource.referralPath': 'unknown'}
B=A.fillna(value=replace_null_values)

# excluded_features = [
#     'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue',
#     'visitId', 'visitStartTime', 'vis_date'
# ]

categorical_features = [
    _f for _f in B.columns
    if (B[_f].dtype == 'object')
]

#print(categorical_features)

for f in categorical_features:
    B[f], indexer = pd.factorize(B[f])

X=B.drop(['shops or not','date','vis_date','totals.transactionRevenue'],axis=1)
y=B['shops or not']
# y=train_df['shops or not']

# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))

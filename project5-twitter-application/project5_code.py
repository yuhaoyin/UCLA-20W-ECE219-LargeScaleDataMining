#!/usr/bin/env python
# coding: utf-8

# ## PART 1 - Popularity Prediction

# <font size=4> **Question 1**: Report the following statistics for each hashtag, i.e. each file:
#   - Average number of tweets per hour;
#   - Average number of followers of users posting the tweets per tweet (to make it simple, we average over the number of tweets; if a user posted twice, we count the user and the user’s followers twice as well);
#   - Average number of retweets per tweet.
# </font>

# In[3]:


import json
import numpy as np

def report_statistics(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        max_time = 0
        min_time = np.inf
        total_followers = 0
        total_retweets = 0
        total_tweets = len(lines)
        for line in lines:
            json_obj = json.loads(line)
            if json_obj['citation_date'] > max_time:
                max_time = json_obj['citation_date']
            if json_obj['citation_date'] < min_time:
                min_time = json_obj['citation_date']
            total_followers += json_obj['author']['followers']
            total_retweets += json_obj['metrics']['citations']['total']
        avg_tweets_per_h = total_tweets * 3600 / (max_time - min_time)
        avg_followers_per_tweet = total_followers / total_tweets
        avg_retweets_per_tweet = total_retweets / total_tweets
        print(filename)
        print('Average number of tweets per hour: ', avg_tweets_per_h)
        print('Average number of followers of users posting the tweets per tweet: ', avg_followers_per_tweet)
        print('Average number of retweets per tweet: ', avg_retweets_per_tweet)
        print('-' * 50)


# In[4]:


files = ['ECE219_tweet_data/tweets_#gohawks.txt', 'ECE219_tweet_data/tweets_#gopatriots.txt', 
         'ECE219_tweet_data/tweets_#nfl.txt', 'ECE219_tweet_data/tweets_#patriots.txt', 
         'ECE219_tweet_data/tweets_#sb49.txt', 'ECE219_tweet_data/tweets_#superbowl.txt']

for file in files:
    report_statistics(file)


# <font size=4> **Question 2:** Plot “number of tweets in hour” over time for #SuperBowl and #NFL (a bar plot with 1-hour bins). The tweets are stored in separate files for different hashtags and files are named as tweet_[#hashtag].txt. </font>

# In[6]:


import math 
import matplotlib.pyplot as plt
import datetime
import pytz

pst_tz = pytz.timezone('America/Los_Angeles')

def report_tweets(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        max_time = 0
        min_time = np.inf
        total_tweets = len(lines)
        for line in lines:
            json_obj = json.loads(line)
            if json_obj['citation_date'] > max_time:
                max_time = json_obj['citation_date']
            if json_obj['citation_date'] < min_time:
                min_time = json_obj['citation_date']
        
        total_hours = math.ceil((max_time - min_time) / 3600)
        n_tweets = [0] * total_hours
        for line in lines:
            json_obj = json.loads(line)
            index = math.floor((json_obj['citation_date'] - min_time) / 3600)
            n_tweets[index] += 1
        return n_tweets


# In[7]:


q2_files = ['ECE219_tweet_data/tweets_#nfl.txt','ECE219_tweet_data/tweets_#superbowl.txt']

for file in q2_files:
    n_tweets = report_tweets(file)
    plt.figure(figsize=(10,6))
    plt.bar(range(len(n_tweets)),n_tweets)
    plt.xlabel('Hours over time')
    plt.ylabel('Number of tweets')
    plt.title('number of tweets in hours for '+file)


# <font size=4> **Question 3:** For each of your models, report your model's Mean Squared Error (MSE) and R-squared measure. Also, analyse the significance of each feature using the t-test and p-value. You may use the OLS in the library statsmodels in Python. </font>

# In[8]:


def report_features(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        max_time = 0
        min_time = np.inf
        total_tweets = len(lines)
        total_followers = 0
        total_retweets = 0
        for line in lines:
            json_obj = json.loads(line)
            if json_obj['citation_date'] > max_time:
                max_time = json_obj['citation_date']
            if json_obj['citation_date'] < min_time:
                min_time = json_obj['citation_date']
            total_followers += json_obj['author']['followers']
            total_retweets += json_obj['metrics']['citations']['total']
        
        total_hours = math.ceil((max_time - min_time) / 3600)
        #initialize features
        features = np.zeros((total_hours,5))
        for hour in range(total_hours):
            features[hour][4] = datetime.datetime.fromtimestamp((min_time + hour * 3600), pst_tz).hour #time of the day
        for line in lines:
            json_obj = json.loads(line)
            index = math.floor((json_obj['citation_date'] - min_time) / 3600)
            features[index][0] += 1 #number of tweets
            features[index][1] += json_obj['metrics']['citations']['total'] #number of retweets
            features[index][2] += json_obj['author']['followers'] #number of followers
            features[index][3] = max(features[index][3], json_obj['author']['followers']) #max number of followers
            
        return features


# In[12]:


import statsmodels.api as sm
from sklearn import metrics

for file in files:
    features = report_features(file)
    x = features[:-1,:]
    y_true = features[1:,0]

    lr_fit = sm.OLS(y_true,x).fit()
    y_pred = lr_fit.predict()
    print('Hashtag: ' + file)
    print('MSE: ', metrics.mean_squared_error(y_true, y_pred))
    print(lr_fit.summary())
    print('\n')


# <font size=4> **Question 4:** Design a regression model using any features from the papers you find or other new features you may find useful for this problem. Fit your model on the data of each hashtag and report fitting MSE and significance of features.

# <font size=4> **Question 5:** For each of the top 3 features (i.e. with the smallest p-values) in your measurements, draw a scatter plot of predictant (number of tweets for next hour) versus value of that feature, using all the samples you have extracted, and analyze it.

# In[ ]:


# find the intersected time intervals for all twitter data
def get_time_interval(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        max_time = 0
        min_time = np.inf
        for line in lines:
            json_obj = json.loads(line)
            if json_obj['citation_date'] > max_time:
                max_time = json_obj['citation_date']
            if json_obj['citation_date'] < min_time:
                min_time = json_obj['citation_date']
        return max_time, min_time
    
max_t = []
min_t = []
for file in files:
    max_time, min_time = get_time_interval(file)
    max_t.append(max_time)
    min_t.append(min_time)

max_time_agg = min(max_t)
min_time_agg = max(min_t)


# In[98]:


feature_names = ['Number of tweets', 'Total number of retweets', 'Sum of the number of followers', 
                'Maximum number of followers', 'Time of the day', 'Sum of ranking score',
                'Sum of passivity', 'Total number of unique users','Total number of unique authors', 
                'Total number of user mentions']

mnth_to_int = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def get_days(user_create_time, tweet_create_time):
    user_create_date = user_create_time.split(' ')
    tweet_create_date = tweet_create_time.split(' ')
    user_create_date = datetime.datetime(int(user_create_date[-1]), mnth_to_int[user_create_date[1]], int(user_create_date[2]))
    tweet_create_date = datetime.datetime(int(tweet_create_date[-1]), mnth_to_int[tweet_create_date[1]], int(tweet_create_date[2]))
    created_days = tweet_create_date - user_create_date
    created_days = created_days.days
    return created_days
    

def report_features2(filename, min_time, max_time):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        total_hours = math.ceil((max_time - min_time) / 3600)
        user_ids = {hour:set() for hour in range(total_hours)}
        author_nicks = {hour:set() for hour in range(total_hours)}
        
        #initialize features
        features = np.zeros((total_hours,len(feature_names)))
        
        for hour in range(total_hours):
            features[hour][4] = datetime.datetime.fromtimestamp((min_time + hour * 3600), pst_tz).hour
            
        for line in lines:
            json_obj = json.loads(line)
            
            if json_obj['citation_date'] >= min_time and json_obj['citation_date'] <= max_time:
                index = math.floor((json_obj['citation_date'] - min_time) / 3600)
                features[index][0] += 1 #number of tweets
                features[index][1] += json_obj['metrics']['citations']['total'] #number of retweets
                features[index][2] += json_obj['author']['followers'] #number of followers
                features[index][3] = max(features[index][3], json_obj['author']['followers']) #max number of followers

                features[index][5] += json_obj['metrics']['ranking_score'] #sum of ranking score
                n_days = get_days(json_obj['tweet']['user']['created_at'], json_obj['tweet']['created_at']) #get the number of days since the user account was created
                features[index][6] += n_days/(1.0 + json_obj['tweet']['user']['statuses_count']) #sum of passivity
                if json_obj['tweet']['user']['id'] not in user_ids[index]:
                    user_ids[index].add(json_obj['tweet']['user']['id'])               
                features[index][7] = len(user_ids[index]) #number of unique users
                if json_obj['author']['nick'] not in author_nicks[index]:
                    author_nicks[index].add(json_obj['author']['nick'])
                features[index][8] = len(author_nicks[index]) #number of unique authors
                features[index][9] += len(json_obj['tweet']['entities']['user_mentions']) #number of user mentions
        
        return features

def scatter_plot(features, hashtag, y_pred, pvalues, feature_names):
    ranked_index = np.argsort(pvalues)
    print('Hashtag: ' + hashtag)
    for i in range(3):
        plt.figure(figsize = (8,5))
        plt.scatter(features[:,ranked_index[i]], y_pred, alpha=0.5)
        plt.xlabel(feature_names[ranked_index[i]])
        plt.ylabel("Number of tweets next hour")
        plt.grid(True)
        plt.show()
    print('-' * 80)


# In[99]:


import statsmodels.api as sm
from sklearn import metrics


for file in files:
    features = report_features2(file, min_time_agg, max_time_agg)
    x = features[:-1,:] #training features
    y_true = features[1:,0] #true labels

    lr_fit = sm.OLS(y_true,x).fit()
    y_pred = lr_fit.predict()
    pvalues = lr_fit.pvalues
    print('Hashtag: ' + file)
    print('MSE: ', metrics.mean_squared_error(y_true, y_pred))
    print(lr_fit.summary())
    scatter_plot(x, file, y_pred, pvalues, feature_names)
    print('\n')


# <font size=4> **Question 6:** We define three time periods and their corresponding window length as follows:
#     1. Before Feb. 1, 8:00 a.m.: 1-hour window
#     2. Between Feb. 1, 8:00 a.m. and 8:00 p.m.: 5-minute window
#     3. After Feb. 1, 8:00 p.m.: 1-hour window
#     
# For each hashtag, train 3 regression models, one for each of these time periods (the times are all in PST). Report the MSE and R-squared score for each case.

# In[78]:


# find the intersected time intervals for all twitter data
def get_time_interval(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        max_time = 0
        min_time = np.inf
        for line in lines:
            json_obj = json.loads(line)
            if json_obj['citation_date'] > max_time:
                max_time = json_obj['citation_date']
            if json_obj['citation_date'] < min_time:
                min_time = json_obj['citation_date']
        return max_time, min_time
    
max_t = []
min_t = []
for file in files:
    max_time, min_time = get_time_interval(file)
    max_t.append(max_time)
    min_t.append(min_time)

max_time_agg = min(max_t)
min_time_agg = max(min_t)


# In[6]:


import datetime
import time
import pytz

pst_tz = pytz.timezone('America/Los_Angeles')

pre_active_time = datetime.datetime(2015, 2, 1, 8, 0, 0, 0, pst_tz)
post_active_time = datetime.datetime(2015, 2, 1, 20, 0, 0, 0, pst_tz)
pre_active_timestamp = time.mktime(pre_active_time.timetuple())
post_active_timestamp = time.mktime(post_active_time.timetuple())


# In[89]:


def report_features3(filename, min_time, max_time):
    with open(filename, 'r') as file:
        lines = file.readlines()
        total_tweets = len(lines)
        total_followers = 0
        total_retweets = 0
    
        min_time = time.mktime(datetime.datetime.fromtimestamp(min_time).replace(minute=0, second=0, microsecond=0).timetuple())
        max_time = time.mktime(datetime.datetime.fromtimestamp(max_time).replace(minute=0, second=0, microsecond=0).timetuple())
        
        time_window_len1 = math.ceil((pre_active_timestamp - min_time) / (60*60)) #Before Feb. 1, 8:00 a.m.: 1-hour window
        time_window_len2 = math.ceil((post_active_timestamp - pre_active_timestamp) / (60*5)) #Between Feb. 1, 8:00 a.m. and 8:00 p.m.: 5-minute window
        time_window_len3 = math.ceil((max_time - post_active_timestamp) / (60*60)) #After Feb. 1, 8:00 p.m.: 1-hour window
        
        total_window_len = time_window_len1 + time_window_len2 + time_window_len3
        user_ids = {time:set() for time in range(total_window_len)}
        author_nicks = {time:set() for time in range(total_window_len)}
        
        #initialize features
        features = np.zeros((total_window_len,len(feature_names)))
        
        #assign features
        for hour1 in range(time_window_len1):
            features[hour1][4] = datetime.datetime.fromtimestamp((min_time + hour1*60*60), pst_tz).hour
            
        for five_minutes in range(time_window_len2):
            features[time_window_len1+five_minutes][4] = datetime.datetime.fromtimestamp((pre_active_timestamp + five_minutes*60*5), pst_tz).hour

        for hour2 in range(time_window_len3):
            features[time_window_len1+time_window_len2+hour2][4] = datetime.datetime.fromtimestamp((post_active_timestamp + hour2*60*60), pst_tz).hour
            
        for line in lines:
            json_obj = json.loads(line)
            if json_obj['citation_date'] >= min_time and json_obj['citation_date'] <= max_time:
                if json_obj['citation_date'] < pre_active_timestamp:
                    index = math.floor((json_obj['citation_date'] - min_time) / 3600)
                    features[index][0] += 1 #number of tweets
                    features[index][1] += json_obj['metrics']['citations']['total'] #number of retweets
                    features[index][2] += json_obj['author']['followers'] #number of followers
                    features[index][3] = max(features[index][3], json_obj['author']['followers']) #max number of followers

                    features[index][5] += json_obj['metrics']['ranking_score'] #sum of ranking score
                    n_days = get_days(json_obj['tweet']['user']['created_at'], json_obj['tweet']['created_at']) #get the number of days since the user account was created
                    features[index][6] += n_days/(1.0 + json_obj['tweet']['user']['statuses_count']) #sum of passivity
                    if json_obj['tweet']['user']['id'] not in user_ids[index]:
                        user_ids[index].add(json_obj['tweet']['user']['id'])               
                    features[index][7] = len(user_ids[index]) #number of unique users
                    if json_obj['author']['nick'] not in author_nicks[index]:
                        author_nicks[index].add(json_obj['author']['nick'])
                    features[index][8] = len(author_nicks[index]) #number of unique authors
                    features[index][9] += len(json_obj['tweet']['entities']['user_mentions']) #number of user mentions

                elif json_obj['citation_date'] < post_active_timestamp:
                    index = math.floor((json_obj['citation_date'] - pre_active_timestamp) / 300) + time_window_len1
                    features[index][0] += 1 #number of tweets
                    features[index][1] += json_obj['metrics']['citations']['total'] #number of retweets
                    features[index][2] += json_obj['author']['followers'] #number of followers
                    features[index][3] = max(features[index][3], json_obj['author']['followers']) #max number of followers

                    features[index][5] += json_obj['metrics']['ranking_score'] #sum of ranking score
                    n_days = get_days(json_obj['tweet']['user']['created_at'], json_obj['tweet']['created_at']) #get the number of days since the user account was created
                    features[index][6] += n_days/(1.0 + json_obj['tweet']['user']['statuses_count']) #sum of passivity
                    if json_obj['tweet']['user']['id'] not in user_ids[index]:
                        user_ids[index].add(json_obj['tweet']['user']['id'])               
                    features[index][7] = len(user_ids[index]) #number of unique users
                    if json_obj['author']['nick'] not in author_nicks[index]:
                        author_nicks[index].add(json_obj['author']['nick'])
                    features[index][8] = len(author_nicks[index]) #number of unique authors
                    features[index][9] += len(json_obj['tweet']['entities']['user_mentions']) #number of user mentions

                else:
                    index = math.floor((json_obj['citation_date'] - post_active_timestamp) / 3600) + time_window_len1 + time_window_len2
                    features[index][0] += 1 #number of tweets
                    features[index][1] += json_obj['metrics']['citations']['total'] #number of retweets
                    features[index][2] += json_obj['author']['followers'] #number of followers
                    features[index][3] = max(features[index][3], json_obj['author']['followers']) #max number of followers

                    features[index][5] += json_obj['metrics']['ranking_score'] #sum of ranking score
                    n_days = get_days(json_obj['tweet']['user']['created_at'], json_obj['tweet']['created_at']) #get the number of days since the user account was created
                    features[index][6] += n_days/(1.0 + json_obj['tweet']['user']['statuses_count']) #sum of passivity
                    if json_obj['tweet']['user']['id'] not in user_ids[index]:
                        user_ids[index].add(json_obj['tweet']['user']['id'])               
                    features[index][7] = len(user_ids[index]) #number of unique users
                    if json_obj['author']['nick'] not in author_nicks[index]:
                        author_nicks[index].add(json_obj['author']['nick'])
                    features[index][8] = len(author_nicks[index]) #number of unique authors
                    features[index][9] += len(json_obj['tweet']['entities']['user_mentions']) #number of user mentions
        
        return features[:time_window_len1], features[time_window_len1:time_window_len1 + time_window_len2], features[time_window_len1 + time_window_len2:]


# In[93]:


for file in files:
    print('Hashtag: ' + file)
    features_before, features_between, features_after = report_features3(file, min_time_agg, max_time_agg)

    x_before = features_before[:-1,:] #training features
    y_true_before = features_before[1:,0] #true labels
    lr_before_fit = sm.OLS(y_true_before,x_before).fit()
    y_pred_before = lr_before_fit.predict()
    print('Before Feb. 1, 8:00 a.m.:')
    print('MSE: ', metrics.mean_squared_error(y_true_before, y_pred_before))
    print(lr_before_fit.summary())
    print('-' * 80)

    x_between = features_between[:-1,:] #training features
    y_true_between = features_between[1:,0] #true labels
    lr_between_fit = sm.OLS(y_true_between,x_between).fit()
    y_pred_between = lr_between_fit.predict()
    print('Between Feb. 1, 8:00 a.m. and 8:00 p.m.:')
    print('MSE: ', metrics.mean_squared_error(y_true_between, y_pred_between))
    print(lr_between_fit.summary())
    print('-' * 80)

    x_after = features_after[:-1,:] #training features
    y_true_after = features_after[1:,0] #true labels
    lr_after_fit = sm.OLS(y_true_after,x_after).fit()
    y_pred_after = lr_after_fit.predict()
    print('After Feb. 1, 8:00 p.m.:')
    print('MSE: ', metrics.mean_squared_error(y_true_after, y_pred_after))
    print(lr_after_fit.summary())

    print('=' * 80)
    print('\n')


# <font size=4> **Question 7:** Also, aggregate the data of all hashtags, and train 3 models (for the intervals mentioned above) to predict the number of tweets in the next time window on the aggregated data. Perform the same evaluations on your combined model and compare with models you trained for individual hashtags.

# In[76]:


# aggregrate features
features_before, features_between, features_after = report_features3(files[0], min_time_agg, max_time_agg)

for file in files[1:]:
    f_before, f_between, f_after = report_features3(file, min_time_agg, max_time_agg)
    features_before[:,:3] += f_before[:,:3]
    features_before[:,3] = np.maximum(f_before[:,3], features_before[:,3])
    features_before[:,5:] += f_before[:,5:]
    
    features_between[:,:3] += f_between[:,:3]
    features_between[:,3] = np.maximum(f_between[:,3], features_between[:,3])
    features_between[:,5:] += f_between[:,5:]
    
    features_after[:,:3] += f_after[:,:3]
    features_after[:,3] = np.maximum(f_after[:,3], features_after[:,3])
    features_after[:,5:] += f_after[:,5:]


# In[77]:


import statsmodels.api as sm
from sklearn import metrics

print('Aggregate data')
x_before = features_before[:-1,:] #training features
y_true_before = features_before[1:,0] #true labels
lr_before_fit = sm.OLS(y_true_before,x_before).fit()
y_pred_before = lr_before_fit.predict()
print('Before Feb. 1, 8:00 a.m.:')
print('MSE: ', metrics.mean_squared_error(y_true_before, y_pred_before))
print(lr_before_fit.summary())
print('-' * 100)
print('\n')

x_between = features_between[:-1,:] #training features
y_true_between = features_between[1:,0] #true labels
lr_between_fit = sm.OLS(y_true_between,x_between).fit()
y_pred_between = lr_between_fit.predict()
print('Between Feb. 1, 8:00 a.m. and 8:00 p.m.:')
print('MSE: ', metrics.mean_squared_error(y_true_between, y_pred_between))
print(lr_between_fit.summary())
print('-' * 100)
print('\n')

x_after = features_after[:-1,:] #training features
y_true_after = features_after[1:,0] #true labels
lr_after_fit = sm.OLS(y_true_after,x_after).fit()
y_pred_after = lr_after_fit.predict()
print('After Feb. 1, 8:00 p.m.:')
print('MSE: ', metrics.mean_squared_error(y_true_after, y_pred_after))
print(lr_after_fit.summary())


# <font size=4> **Question 8:** Use grid search to find the best parameter set for RandomForestRegressor and GradientBoostingRegressor respectively. Use the following param grid
#     
#     {
#     'max_depth': [10, 30, 50, 70, 100, 200, None], 
#     'max_features': ['auto', 'sqrt'], 
#     'min_samples_leaf': [1, 2,3, 4], 
#     'min_samples_split': [2, 5, 10], 
#     'n_estimators': [200, 400, 600, 800, 1000,
#                     1200, 1400, 1600, 1800, 2000]
#     }
# Set cv = KFold(5, shuffle=True), scoring=’neg_mean_squared_error’ for the grid search.

# In[119]:


# aggregrate entire dataset
data_agg = report_features2(files[0], min_time_agg, max_time_agg)

for file in files[1:]:
    data = report_features2(file, min_time_agg, max_time_agg)
    data_agg[:,:3] += data[:,:3]
    data_agg[:,3] = np.maximum(data[:,3], data_agg[:,3])
    data_agg[:,5:] += data[:,5:]
    
x_agg = data_agg[:-1,:]
y_agg = data_agg[1:,0]


# In[120]:


# RandomForest GridSearch
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


pipe_rf = Pipeline([
    ('standardize', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}

grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                       scoring='neg_mean_squared_error').fit(x_agg, y_agg)
result_rf = pd.DataFrame(grid_rf.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                             'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                             'param_model__n_estimators']]
result_rf = result_rf.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_rf.head()


# In[121]:


# GradientBoosting GridSearch
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


pipe_gb = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}


grid_gb = GridSearchCV(pipe_gb, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                       scoring='neg_mean_squared_error').fit(x_agg, y_agg)
result_gb = pd.DataFrame(grid_gb.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                             'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                             'param_model__n_estimators']]
result_gb = result_gb.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_gb.head()


# <font size=4> **Question 10:** For each time period described in **Question 6**, perform the same grid search above for GradientBoostingRegressor (with corresponding time window length). Does the cross-validation test error change? Are the best parameter set you find in each period agree with those you found above?

# In[122]:


# PreActive: Before Feb. 1, 8 a.m.
x_agg_before = features_before[:-1,:]
y_agg_before = features_before[1:,0]


pipe_before_gb = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}


grid_before_gb = GridSearchCV(pipe_before_gb, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                              scoring='neg_mean_squared_error').fit(x_agg_before, y_agg_before)
result_before_gb = pd.DataFrame(grid_before_gb.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                                             'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                                             'param_model__n_estimators']]
result_before_gb = result_before_gb.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_before_gb.head()


# In[123]:


# Active: Between Feb.1, 8 a.m. and 8 p.m.
x_agg_between = features_between[:-1,:]
y_agg_between = features_between[1:,0]


pipe_between_gb = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}


grid_between_gb = GridSearchCV(pipe_between_gb, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                              scoring='neg_mean_squared_error').fit(x_agg_between, y_agg_between)
result_between_gb = pd.DataFrame(grid_between_gb.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                                             'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                                             'param_model__n_estimators']]
result_between_gb = result_between_gb.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_between_gb.head()


# In[124]:


#PostActive: After Feb.1, 8 p.m.
x_agg_after = features_after[:-1,:]
y_agg_after = features_after[1:,0]


pipe_after_gb = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}


grid_after_gb = GridSearchCV(pipe_after_gb, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                              scoring='neg_mean_squared_error').fit(x_agg_after, y_agg_after)
result_after_gb = pd.DataFrame(grid_after_gb.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                                           'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                                           'param_model__n_estimators']]
result_after_gb = result_after_gb.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_after_gb.head()


# <font size=4> **Question 11:** Now try to regress the aggregated data with MLPRegressor. Try different architectures (i.e. the structure of the network) by adjusting hidden layer sizes. You should try at least 5 architectures with various numbers of layers and layer sizes. Report the architectures you tried, as well as its MSE of fitting the entire aggregated data.

# In[128]:


# NeuralNetwork GridSearch

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

pipe_nn_noscale = Pipeline([
    ('model', MLPRegressor(random_state=42, max_iter=2000))
])

param_grid = {
    'model__hidden_layer_sizes': [(x,y) for x in np.arange(1, 101) for y in np.arange(1, 101)]
}


grid_nn_noscale = GridSearchCV(pipe_nn_noscale, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                               scoring='neg_mean_squared_error').fit(x_agg, y_agg)
result_nn_noscale = pd.DataFrame(grid_nn_noscale.cv_results_)[['mean_test_score', 'param_model__hidden_layer_sizes']]
result_nn_noscale = result_nn_noscale.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_nn_noscale.head()


# <font size=4> **Question 12:**  Use StandardScaler to scale the features before feeding it to MLPRegressor (with the best architecture you got above). Does its performance increase?

# In[130]:


import numpy as np
from sklearn.model_selection import cross_validate

pipe_nn_scale = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(hidden_layer_sizes=(11, 15), random_state=42, max_iter=2000))
])

score_nn_scale = cross_validate(pipe_nn_scale, x_agg, y_agg, scoring='neg_mean_squared_error', 
                                cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=0)

print('Neural Network | MSE after Feature Standardization:', -np.mean(score_nn_scale['test_score']))


# <font size=4> **Question 13:** Using grid search, find the best architecture (for scaled data) for each period (with corresponding window length) described in **Question 6**.

# In[131]:


# PreActive: Before Feb. 1, 8 a.m.
x_agg_before = features_before[:-1,:]
y_agg_before = features_before[1:,0]


pipe_before_nn = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(random_state=42, max_iter=2000))
])


param_grid = {
    'model__hidden_layer_sizes': [(x,y) for x in np.arange(1, 101) for y in np.arange(1, 101)]
}



grid_before_nn = GridSearchCV(pipe_before_nn, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                              scoring='neg_mean_squared_error').fit(x_agg_before, y_agg_before)
result_before_nn = pd.DataFrame(grid_before_nn.cv_results_)[['mean_test_score', 'param_model__hidden_layer_sizes']]
result_before_nn = result_before_nn.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_before_nn.head()


# In[132]:


# Active: Between Feb.1, 8 a.m. and 8 p.m.
x_agg_between = features_between[:-1,:]
y_agg_between = features_between[1:,0]


pipe_between_nn = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(random_state=42, max_iter=2000))
])

param_grid = {
    'model__hidden_layer_sizes': [(x,y) for x in np.arange(1, 101) for y in np.arange(1, 101)]
}


grid_between_nn = GridSearchCV(pipe_between_nn, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                              scoring='neg_mean_squared_error').fit(x_agg_between, y_agg_between)
result_between_nn = pd.DataFrame(grid_between_nn.cv_results_)[['mean_test_score', 'param_model__hidden_layer_sizes']]
result_between_nn = result_between_nn.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_between_nn.head()


# In[133]:


#PostActive: After Feb.1, 8 p.m.
x_agg_after = features_after[:-1,:]
y_agg_after = features_after[1:,0]


pipe_after_nn = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(random_state=42, max_iter=2000))
])

param_grid = {
    'model__hidden_layer_sizes': [(x,y) for x in np.arange(1, 101) for y in np.arange(1, 101)]
}


grid_after_nn = GridSearchCV(pipe_after_nn, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                              scoring='neg_mean_squared_error').fit(x_agg_after, y_agg_after)
result_after_nn = pd.DataFrame(grid_after_nn.cv_results_)[['mean_test_score', 'param_model__hidden_layer_sizes']]
result_after_nn = result_after_nn.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_after_nn.head()


# <font size=4> **Question 14:** Use 6x window to predict. Report the model you use. For each test file, provide your predictions on the number of tweets in the next time window.

# In[230]:


def report_features_6x_window(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        max_time = 0
        min_time = np.inf
        total_tweets = len(lines)
        total_followers = 0
        total_retweets = 0
    
        for line in lines:
            json_obj = json.loads(line)
            if json_obj['citation_date'] > max_time:
                max_time = json_obj['citation_date']
            if json_obj['citation_date'] < min_time:
                min_time = json_obj['citation_date']
        
        if min_time > pre_active_timestamp and max_time < post_active_timestamp:
            # Active Time Period
            total_five_minutes = math.ceil((max_time - min_time) / (60*5))
            user_ids = {five_minute:set() for five_minute in range(total_five_minutes)}
            author_nicks = {five_minute:set() for five_minute in range(total_five_minutes)}
            
            features = np.zeros((total_five_minutes,len(feature_names)))
            for five_minute in range(total_five_minutes):
                features[five_minute][4] = datetime.datetime.fromtimestamp((min_time + five_minute * 300), pst_tz).hour
            for line in lines:
                json_obj = json.loads(line)
                index = math.floor((json_obj['citation_date'] - min_time) / 300)
                features[index][0] += 1 #number of tweets
                features[index][1] += json_obj['metrics']['citations']['total'] #number of retweets
                features[index][2] += json_obj['author']['followers'] #number of followers
                features[index][3] = max(features[index][3], json_obj['author']['followers']) #max number of followers

                features[index][5] += json_obj['metrics']['ranking_score'] #sum of ranking score
                n_days = get_days(json_obj['tweet']['user']['created_at'], json_obj['tweet']['created_at']) #get the number of days since the user account was created
                features[index][6] += n_days/(1.0 + json_obj['tweet']['user']['statuses_count']) #sum of passivity
                if json_obj['tweet']['user']['id'] not in user_ids[index]:
                    user_ids[index].add(json_obj['tweet']['user']['id'])               
                features[index][7] = len(user_ids[index]) #number of unique users
                if json_obj['author']['nick'] not in author_nicks[index]:
                    author_nicks[index].add(json_obj['author']['nick'])
                features[index][8] = len(author_nicks[index]) #number of unique authors
                features[index][9] += len(json_obj['tweet']['entities']['user_mentions']) #number of user mentions
                
        else:
            total_hours = math.ceil((max_time - min_time) / 3600)
            user_ids = {hour:set() for hour in range(total_hours)}
            author_nicks = {hour:set() for hour in range(total_hours)}
            
            features = np.zeros((total_hours,len(feature_names)))
            for hour in range(total_hours):
                features[hour][4] = datetime.datetime.fromtimestamp((min_time + hour * 3600), pst_tz).hour
            for line in lines:
                json_obj = json.loads(line)
                index = math.floor((json_obj['citation_date'] - min_time) / 3600)
                features[index][0] += 1 #number of tweets
                features[index][1] += json_obj['metrics']['citations']['total'] #number of retweets
                features[index][2] += json_obj['author']['followers'] #number of followers
                features[index][3] = max(features[index][3], json_obj['author']['followers']) #max number of followers

                features[index][5] += json_obj['metrics']['ranking_score'] #sum of ranking score
                n_days = get_days(json_obj['tweet']['user']['created_at'], json_obj['tweet']['created_at']) #get the number of days since the user account was created
                features[index][6] += n_days/(1.0 + json_obj['tweet']['user']['statuses_count']) #sum of passivity
                if json_obj['tweet']['user']['id'] not in user_ids[index]:
                    user_ids[index].add(json_obj['tweet']['user']['id'])               
                features[index][7] = len(user_ids[index]) #number of unique users
                if json_obj['author']['nick'] not in author_nicks[index]:
                    author_nicks[index].add(json_obj['author']['nick'])
                features[index][8] = len(author_nicks[index]) #number of unique authors
                features[index][9] += len(json_obj['tweet']['entities']['user_mentions']) #number of user mentions
        
        features_6x_window = np.zeros(shape=[1, features.shape[1]])
        features_6x_window[0, :3] = np.sum(features[:, :3], axis=0)
        features_6x_window[0, 3] = np.max(features[:, 3])
        features_6x_window[0, 4] = np.mean(features[:, 4])
        features_6x_window[0, 5:] = np.sum(features[:, 5:], axis=0)
        
        return features_6x_window


# In[231]:


test_files = ['ECE219_tweet_test/sample0_period1.txt', 'ECE219_tweet_test/sample0_period2.txt', 'ECE219_tweet_test/sample0_period3.txt',
             'ECE219_tweet_test/sample1_period1.txt', 'ECE219_tweet_test/sample1_period2.txt', 'ECE219_tweet_test/sample1_period3.txt',
             'ECE219_tweet_test/sample2_period1.txt', 'ECE219_tweet_test/sample2_period2.txt', 'ECE219_tweet_test/sample2_period3.txt']

x_test_period1 = np.zeros(shape=[3, len(feature_names)])
x_test_period2 = np.zeros(shape=[3, len(feature_names)])
x_test_period3 = np.zeros(shape=[3, len(feature_names)])

for i in range(len(test_files)):
    if i % 3 == 0:
        x_test_period1[i//3, :] = report_features_6x_window(test_files[i])
    elif i % 3 == 1:
        x_test_period2[i//3, :] = report_features_6x_window(test_files[i])
    else:
        x_test_period3[i//3, :] = report_features_6x_window(test_files[i])


# In[185]:


# PreActive: calculate new features for 6x window and GradientBoosting GridSearch
x_agg_before_6x_window = np.zeros(shape=[features_before.shape[0]-6, len(feature_names)])

for i in range(features_before.shape[0] - 6):
    x_agg_before_6x_window[i, :3] = np.sum(features_before[i:(i+6), :3], axis=0) # sum of number of tweets, retweets, and followers
    x_agg_before_6x_window[i, 3] = np.max(features_before[i:(i+6), 3]) # maximum number of followers
    x_agg_before_6x_window[i, 4] = np.max(features_before[i:(i+6), 4]) # inter-median time of the day
    x_agg_before_6x_window[i, 5:] = np.sum(features_before[i:(i+6), 5:], axis=0)
    
y_agg_before_6x_window = features_before[6:, 0]


pipe_before_6x_window_gb = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}


grid_before_6x_window_gb = GridSearchCV(pipe_before_6x_window_gb, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                                        scoring='neg_mean_squared_error').fit(x_agg_before_6x_window, y_agg_before_6x_window)
result_before_6x_window_gb = pd.DataFrame(grid_before_6x_window_gb.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                                                                'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                                                                'param_model__n_estimators']]
result_before_6x_window_gb = result_before_6x_window_gb.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_before_6x_window_gb.head()


# In[241]:


pipe_before_6x_window_gb_optim = Pipeline([
        ('standardize', StandardScaler()),
        ('model', GradientBoostingRegressor(max_depth=10, max_features='auto', min_samples_leaf=1,
                                            min_samples_split=10, n_estimators=200, random_state=42))
])

pipe_before_6x_window_gb_optim.fit(x_agg_before_6x_window, y_agg_before_6x_window)
print('Predictions for Period 1:', pipe_before_6x_window_gb_optim.predict(x_test_period1))


# In[187]:


# Active: calculate new features for 6x window and GradientBoosting GridSearch
x_agg_between_6x_window = np.zeros(shape=[features_between.shape[0]-6, len(feature_names)])

for i in range(features_between.shape[0] - 6):
    x_agg_between_6x_window[i, :3] = np.sum(features_between[i:(i+6), :3], axis=0) # sum of number of tweets, retweets, and followers
    x_agg_between_6x_window[i, 3] = np.max(features_between[i:(i+6), 3]) # maximum number of followers
    x_agg_between_6x_window[i, 4] = np.max(features_between[i:(i+6), 4]) # inter-median time of the day
    x_agg_between_6x_window[i, 5:] = np.sum(features_between[i:(i+6), 5:], axis=0)
    
y_agg_between_6x_window = features_between[6:, 0]


pipe_between_6x_window_gb = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}


grid_between_6x_window_gb = GridSearchCV(pipe_between_6x_window_gb, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                                        scoring='neg_mean_squared_error').fit(x_agg_between_6x_window, y_agg_between_6x_window)
result_between_6x_window_gb = pd.DataFrame(grid_between_6x_window_gb.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                                                                'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                                                                'param_model__n_estimators']]
result_between_6x_window_gb = result_between_6x_window_gb.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_between_6x_window_gb.head()


# In[240]:


pipe_between_6x_window_gb_optim = Pipeline([
        ('standardize', StandardScaler()),
        ('model', GradientBoostingRegressor(max_depth=50, max_features='auto', min_samples_leaf=3,
                                            min_samples_split=5, n_estimators=1600, random_state=42))
])

pipe_between_6x_window_gb_optim.fit(x_agg_between_6x_window, y_agg_between_6x_window)
print('Predictions for Period 2:', pipe_between_6x_window_gb_optim.predict(x_test_period2))


# In[186]:


# PostActive: calculate new features for 6x window and GradientBoosting GridSearch
x_agg_after_6x_window = np.zeros(shape=[features_after.shape[0]-6, len(feature_names)])

for i in range(features_after.shape[0] - 6):
    x_agg_after_6x_window[i, :3] = np.sum(features_after[i:(i+6), :3], axis=0) # sum of number of tweets, retweets, and followers
    x_agg_after_6x_window[i, 3] = np.max(features_after[i:(i+6), 3]) # maximum number of followers
    x_agg_after_6x_window[i, 4] = np.max(features_after[i:(i+6), 4]) # inter-median time of the day
    x_agg_after_6x_window[i, 5:] = np.sum(features_after[i:(i+6), 5:], axis=0)
    
y_agg_after_6x_window = features_after[6:, 0]


pipe_after_6x_window_gb = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None],
    'model__max_features': ['auto', 'sqrt'],
    'model__min_samples_leaf': [1, 2, 3, 4],
    'model__min_samples_split': [2, 5, 10],
    'model__n_estimators': [200, 400, 600, 800, 1000,
                           1200, 1400, 1600, 1800, 2000]
}


grid_after_6x_window_gb = GridSearchCV(pipe_after_6x_window_gb, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
                                        scoring='neg_mean_squared_error').fit(x_agg_after_6x_window, y_agg_after_6x_window)
result_after_6x_window_gb = pd.DataFrame(grid_after_6x_window_gb.cv_results_)[['mean_test_score', 'param_model__max_depth', 'param_model__max_features',
                                                                                'param_model__min_samples_leaf', 'param_model__min_samples_split',
                                                                                'param_model__n_estimators']]
result_after_6x_window_gb = result_after_6x_window_gb.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_after_6x_window_gb.head()


# In[239]:


pipe_after_6x_window_gb_optim = Pipeline([
        ('standardize', StandardScaler()),
        ('model', GradientBoostingRegressor(max_depth=None, max_features='auto', min_samples_leaf=4,
                                            min_samples_split=5, n_estimators=2000, random_state=42))
])

pipe_after_6x_window_gb_optim.fit(x_agg_after_6x_window, y_agg_after_6x_window)
print('Predictions for Period 3:', pipe_after_6x_window_gb_optim.predict(x_test_period3))


# ## PART 2 - Fan Base Prediction

# <font size=4> **Question 15.1:** Explain the method you use to determine whether the location is in Washington, Massachusetts or neither. Only use the tweets whose authors belong to either Washington or Massachusetts for the next part.

# In[2]:


import json

# exclude any tweets whose author is not from Washington or Massachusetts
WA_substrings = ['Washington', 'Seattle', 'WA']
MA_substrings = ['Massachusetts', 'Boston', 'MA']

superbowl_dataset_trimmed = []

with open('ECE219_tweet_data/tweets_#superbowl.txt', 'r') as file:
    lines = file.readlines()
    
    for line in lines:
        json_obj = json.loads(line)
        location = json_obj['tweet']['user']['location']
        
        for w in WA_substrings:
            if w in location:
                superbowl_dataset_trimmed.append((json_obj['tweet']['text'], 'Washington'))
                break
                
        for m in MA_substrings:
            if m in location:
                superbowl_dataset_trimmed.append((json_obj['tweet']['text'], 'Massachusetts'))
                break


# <font size=4> **Question 15.2:** Train a binary classifier to predict the location of the author of a tweet (Washington or Massachusetts), given only the textual content of the tweet (using the techniques you learnt in project 1). Try different classification algorithms (at least 3). For each, plot ROC curve, report confusion matrix, and calculate accuracy, recall and precision.

# In[46]:


import numpy as np
from sklearn.model_selection import train_test_split

x_superbowl = np.array(superbowl_dataset_trimmed)[:, 0]
y_superbowl = np.array(superbowl_dataset_trimmed)[:, 1]

y_superbowl_binary = np.zeros(y_superbowl.shape)
y_superbowl_binary[y_superbowl == 'Washington'] = 1

x_train, x_test, y_train, y_test = train_test_split(x_superbowl, y_superbowl_binary, test_size=0.1, random_state=42)


# In[4]:


import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict

# pos_tags: treebank to wordnet
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

wnl = WordNetLemmatizer()
    
def lemmatize(data):
    lemmatized = []
    for doc in data:
        tokens = word_tokenize(doc)
        words = [wnl.lemmatize(word, tag_map[tag[0]]) for word,tag in pos_tag(tokens) 
                 if wnl.lemmatize(word, tag_map[tag[0]]).isalpha()]
        sentence = ' '.join(words)
        lemmatized.append(sentence)
    return lemmatized


# In[49]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

x_train_lemmatized = lemmatize(x_train)
x_test_lemmatized = lemmatize(x_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train_lemmatized)
x_test_tfidf = tfidf_vectorizer.transform(x_test_lemmatized)

svd = TruncatedSVD(n_components=50, random_state=42)
x_train_svd = svd.fit_transform(x_train_tfidf)
x_test_svd = svd.transform(x_test_tfidf)


# In[50]:


# Logistic Regression: GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd


grid_logistic = GridSearchCV(estimator=LogisticRegression(random_state=42), 
                        param_grid={'C':[10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3],
                                   'penalty': ['l1', 'l2', 'elasticnet']}, 
                        cv=5, n_jobs=-1, verbose=1).fit(x_train_svd, y_train)

result_logistic = pd.DataFrame(grid_logistic.cv_results_)[['mean_test_score', 'param_C', 'param_penalty']]
result_logistic = result_logistic.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_logistic.head()


# In[51]:


logistic_optim = LogisticRegression(penalty=grid_logistic.best_params_['penalty'], 
                                    C=grid_logistic.best_params_['C'], random_state=42)

logistic_optim.fit(x_train_svd, y_train)


# In[52]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

y_pred_logistic = logistic_optim.predict(x_test_svd)
y_pred_prob_logistic = logistic_optim.predict_proba(x_test_svd)[:,1]

print('Logistic Regression:')
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred_logistic))
print('accuracy:', accuracy_score(y_test, y_pred_logistic))
print('recall:', recall_score(y_test, y_pred_logistic))
print('precision:', precision_score(y_test, y_pred_logistic))
print('f1_score:', f1_score(y_test, y_pred_logistic))


# In[1]:


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

class_names = ['MA', 'WA']

plot_confusion_matrix(logistic_optim, x_test_svd, y_test, display_labels=class_names, 
                             values_format='d', cmap=plt.cm.Greys)
plt.tight_layout()
plt.title('Confusion Matrix for Logistic Regression', fontweight='bold')
plt.show()


# In[63]:


# RandomForest: GridSearch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe_rfc = Pipeline([
    ('standardize', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None]
}


grid_rfc = GridSearchCV(pipe_rfc, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1)
grid_rfc.fit(x_train_svd, y_train)

result_rfc = pd.DataFrame(grid_rfc.cv_results_)[['mean_test_score', 'param_model__max_depth']]
result_rfc = result_rfc.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_rfc.head()


# In[68]:


pipe_rfc_optim = Pipeline([
        ('standardize', StandardScaler()),
        ('model', RandomForestClassifier(max_depth=30, random_state=42))
])

pipe_rfc_optim.fit(x_train_svd, y_train)


# In[69]:


y_pred_rfc = pipe_rfc_optim.predict(x_test_svd)
y_pred_prob_rfc = pipe_rfc_optim.predict_proba(x_test_svd)[:,1]

print('Logistic Regression:')
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred_rfc))
print('accuracy:', accuracy_score(y_test, y_pred_rfc))
print('recall:', recall_score(y_test, y_pred_rfc))
print('precision:', precision_score(y_test, y_pred_rfc))
print('f1_score:', f1_score(y_test, y_pred_rfc))


# In[70]:


plot_confusion_matrix(pipe_rfc_optim, x_test_svd, y_test, display_labels=class_names, 
                             values_format='d', cmap=plt.cm.Greys)
plt.tight_layout()
plt.title('Confusion Matrix for Random Forest', fontweight='bold')
plt.show()


# In[72]:


# GradientBoosting: GridSearch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

pipe_gbc = Pipeline([
    ('standardize', StandardScaler()),
    ('model', GradientBoostingClassifier(random_state=42))
])

param_grid = {
    'model__max_depth': [10, 30, 50, 70, 100, 200, None]
}


grid_gbc = GridSearchCV(pipe_gbc, param_grid=param_grid, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1, verbose=1)
grid_gbc.fit(x_train_svd, y_train)

result_gbc = pd.DataFrame(grid_gbc.cv_results_)[['mean_test_score', 'param_model__max_depth']]
result_gbc = result_gbc.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
result_gbc.head()


# In[74]:


pipe_gbc_optim = Pipeline([
        ('standardize', StandardScaler()),
        ('model', GradientBoostingClassifier(max_depth=10, random_state=42))
])

pipe_gbc_optim.fit(x_train_svd, y_train)


# In[75]:


y_pred_gbc = pipe_gbc_optim.predict(x_test_svd)
y_pred_prob_gbc = pipe_gbc_optim.predict_proba(x_test_svd)[:,1]

print('Logistic Regression:')
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred_gbc))
print('accuracy:', accuracy_score(y_test, y_pred_gbc))
print('recall:', recall_score(y_test, y_pred_gbc))
print('precision:', precision_score(y_test, y_pred_gbc))
print('f1_score:', f1_score(y_test, y_pred_gbc))


# In[76]:


plot_confusion_matrix(pipe_gbc_optim, x_test_svd, y_test, display_labels=class_names, 
                             values_format='d', cmap=plt.cm.Greys)
plt.tight_layout()
plt.title('Confusion Matrix for Gradient Boosting', fontweight='bold')
plt.show()


# In[89]:


# aggregated ROC curves
from sklearn.metrics import roc_curve

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_prob_logistic)
fpr_rfc, tpr_rfc, _ = roc_curve(y_test, y_pred_prob_rfc)
fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_pred_prob_gbc)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_logistic, tpr_logistic, label = 'LogisticRegression', color='b', linewidth=1.5)
plt.plot(fpr_rfc, tpr_rfc, label = 'RandomForest', color='r', linewidth=1.5)
plt.plot(fpr_gbc, tpr_gbc, label = 'GradientBoosting', color='g', linewidth=1.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('ROC Curves for Comparing Three Different Algorithms', weight='bold')
plt.show()


# ## PART 3 - Define Your Own Project

# In[4]:


import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict

# pos_tags: treebank to wordnet
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

wnl = WordNetLemmatizer()
    
def lemmatize_tweet(tweet):
    tokens = word_tokenize(tweet)
    words = [wnl.lemmatize(word, tag_map[tag[0]]) for word,tag in pos_tag(tokens) 
             if wnl.lemmatize(word, tag_map[tag[0]]).isalpha()]
    sentence = ' '.join(words)
    return sentence


# In[9]:


import math
import json
import numpy as np
from textblob import TextBlob

def perform_sentiment_analysis(filename, min_time=pre_active_timestamp, max_time=post_active_timestamp):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        num_time_window = math.ceil((max_time - min_time) / (3600))
        
        # initialization
        sentiment_summary = []
        for _ in range(num_time_window):
            sentiment_summary.append([])
        positive_tweets, negative_tweets = np.zeros((num_time_window, 1)), np.zeros((num_time_window, 1))
            
        for line in lines:
            # retrieve index of one-hour time window
            json_obj = json.loads(line)
            date = json_obj['citation_date']
            if min_time <= date <= max_time:
                index = math.floor((date - min_time) / 3600)
                # sentiment polarity
                tweet = json_obj['tweet']['text']
                polarity = TextBlob(lemmatize_tweet(tweet)).sentiment.polarity
                if polarity > 0:
                    positive_tweets[index, 0] += 1
                elif polarity < 0:
                    negative_tweets[index, 0] += 1

                sentiment_summary[index].append(polarity)
        
        sentiment_dist = []
        for x in sentiment_summary:
            if x:
                sentiment_dist.append(np.mean(x))
            else:
                sentiment_dist.append(sentiment_dist[-1])
        return sentiment_dist, positive_tweets, negative_tweets


# In[10]:


seahawks_sentiment, seahawks_positive, seahawks_negative = perform_sentiment_analysis('ECE219_tweet_data/tweets_#gohawks.txt')
patriots_sentiment, patriots_positive, patriots_negative = perform_sentiment_analysis('ECE219_tweet_data/tweets_#gopatriots.txt')


# In[13]:


plt.plot(np.arange(8, 20), seahawks_sentiment, label='Seahawks', color='b', linewidth=2)
plt.plot(np.arange(8, 20), patriots_sentiment, label='Patriots', color='r', linewidth=2)
plt.xlabel('Hour')
plt.ylabel('Average Sentiment Polarity')
plt.legend()
plt.title('Average Sentiment Polarity Over Time', weight='bold')
plt.show()


# In[18]:


plt.plot(np.arange(8, 20), seahawks_positive, label='Positive', color='b', linewidth=2)
plt.plot(np.arange(8, 20), seahawks_negative, label='Negative', color='r', linewidth=2)
plt.xlabel('Hour')
plt.ylabel('Number of Tweets')
plt.legend()
plt.title('#gohawks Positive and Negative Tweets Over Time', weight='bold')
plt.show()


# In[17]:


plt.plot(np.arange(8, 20), patriots_positive, label='Positive', color='b', linewidth=2)
plt.plot(np.arange(8, 20), patriots_negative, label='Negative', color='r', linewidth=2)
plt.xlabel('Hour')
plt.ylabel('Number of Tweets')
plt.legend()
plt.title('#gopatriots Positive and Negative Tweets Over Time', weight='bold')
plt.show()


# In[ ]:





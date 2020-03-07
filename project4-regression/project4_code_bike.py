#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
bike_sharing_df = pd.read_csv('Bike-Sharing-Dataset/day.csv')
col_names = list(bike_sharing_df.columns)
bike_sharing_df.head()


# In[3]:


BS_df = bike_sharing_df[['casual','registered','cnt']]


# In[4]:


BS_df.head()


# ## Q1
# Plot a heatmap of Pearson correlation matrix of dataset columns. Report which
# features have the highest absolute correlation with the target variable and what
# that implies.

# In[5]:


import seaborn as sns
corr = bike_sharing_df.iloc[:,1:].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, linewidths=0.5,cmap="RdBu_r",annot=True)


# ## Q2
# Plot the histogram of numerical features. What preprocessing can be done if the
# distribution of a feature has high skewness?

# In[6]:


import matplotlib.pyplot as plt

numerical_features = ['temp','atemp','hum','windspeed']
for i in numerical_features:
    print(i)
    plt.hist(bike_sharing_df[i],bins=20, edgecolor='k', facecolor='b', linewidth=1.5, alpha=0.8)
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show()


# In[7]:


import seaborn as sb
categorical_features = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
for i in categorical_features:
    for j in range(1,4):
        sb.boxplot(bike_sharing_df[i], bike_sharing_df.iloc[:,-j], order=list(set(bike_sharing_df[i])))
        plt.title(i +' vs '+ col_names[-j])
        plt.show()


# In[8]:


month_cnt = [bike_sharing_df['cnt'][0]]
months = []
for i in range(1,bike_sharing_df.shape[0]):
    if bike_sharing_df['mnth'][i] == bike_sharing_df['mnth'][i-1]:
        month_cnt.append(bike_sharing_df['cnt'][i])
    else:
        months.append(month_cnt)
        month_cnt = [bike_sharing_df['cnt'][i]]       


# In[9]:


import numpy as np
for i in range(6):
    plt.title('Month ' + str(i+1))
    plt.bar(range(len(months[i])),months[i])
    plt.show()


# ## Q6
# Handling Categorical Features
# 
# Categorical feature in Bike Sharing dataset is dteday, which can be represented as day difference by instant. Therefore, we ingore this categorical feature.

# In[10]:


bs_temp = bike_sharing_df.drop(['instant','dteday'], axis=1)
ys = bs_temp.iloc[:,-1]
bs_temp = bs_temp.iloc[:,:-3]
xs_scalar = bs_temp
weekday_onehot = pd.get_dummies(bs_temp['weekday'],prefix='weekday')
season_onehot = pd.get_dummies(bs_temp['season'],prefix='season')
mnth_onehot = pd.get_dummies(bs_temp['mnth'],prefix='mnth')
weathersit_onehot = pd.get_dummies(bs_temp['weathersit'],prefix='weathersit')

xs_onehot = pd.concat([bs_temp,weekday_onehot,season_onehot,mnth_onehot,weathersit_onehot], axis=1).drop(['weekday','season','mnth','weathersit'],axis=1)
xs_onehot


# In[11]:


xs_scalar


# ## Q7

# In[12]:


from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_regression, f_regression
xs_scalar_standard = preprocessing.scale(xs_scalar)
xs_onehot_standard = preprocessing.scale(xs_onehot)


# In[13]:


# select 5 most important variables 
MutualInfo_scalar = mutual_info_regression(xs_scalar_standard,ys)
Fscore_scalar = f_regression(xs_scalar_standard, ys)
MutualInfo_onehot = mutual_info_regression(xs_onehot_standard,ys)
Fscore_onehot = f_regression(xs_onehot_standard, ys)

top5_MI_scalar = np.argsort(MutualInfo_scalar)[::-1][:5]
top5_FS_scalar = np.argsort(Fscore_scalar[0])[::-1][:5]
top5_MI_onehot = np.argsort(MutualInfo_onehot)[::-1][:5]
top5_FS_onehot = np.argsort(Fscore_onehot[0])[::-1][:5]

xs_top5_MI_scalar = xs_scalar.iloc[:, top5_MI_scalar]
xs_top5_FS_scalar = xs_scalar.iloc[:, top5_FS_scalar]
xs_top5_MI_onehot = xs_onehot.iloc[:, top5_MI_onehot]
xs_top5_FS_onehot = xs_onehot.iloc[:, top5_FS_onehot]


# In[14]:


xs_top5_MI = xs_top5_MI_onehot
xs_top5_FS = xs_top5_FS_onehot


# ## Linear Regression

# In[15]:


from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

pipe_lr_standard = Pipeline([
    ('standardize', preprocessing.StandardScaler()),
    ('model', LinearRegression())
])

pipe_lr_nonstandard = Pipeline([
    ('model', LinearRegression())
])

lr_train_score = []
lr_test_score = []

cv_results = cross_validate(pipe_lr_nonstandard, xs_top5_FS_onehot, ys, scoring='neg_root_mean_squared_error', 
                            return_train_score=True, cv=10)
lr_train_score.append(np.mean(cv_results['train_score']))
lr_test_score.append(np.mean(cv_results['test_score']))
cv_results = cross_validate(pipe_lr_standard, xs_top5_FS_onehot, ys, scoring='neg_root_mean_squared_error', 
                            return_train_score=True, cv=10)
lr_train_score.append(np.mean(cv_results['train_score']))
lr_test_score.append(np.mean(cv_results['test_score']))


# In[16]:


lr_results = pd.DataFrame(data={'mean_test_score': lr_test_score, 'mean_train_score': lr_train_score,
                                'param_model': 'LinearRegression()', 'param_model__alpha': 'NA', 
                                'Standardize': [False, True]})


# In[17]:


lr_results


# In[18]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso

pipe_standard = Pipeline([
    ('standardize', preprocessing.StandardScaler()),
    ('model', LinearRegression())
])

pipe_nonstandard = Pipeline([
    ('model', LinearRegression())
])

param_grid = {
    'model': [Ridge(random_state=42, max_iter=10000), Lasso(random_state=42, max_iter=10000)],
    'model__alpha': [10.0**x for x in np.arange(-3,4)]
}


# In[19]:


grid1 = GridSearchCV(pipe_nonstandard, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_FS, ys)


# In[20]:


grid2 = GridSearchCV(pipe_standard, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_FS, ys)


# In[21]:


lr_reg_result1 = pd.DataFrame(grid1.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model', 'param_model__alpha']]
lr_reg_result1['Standardize'] = False

lr_reg_result2 = pd.DataFrame(grid2.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model', 'param_model__alpha']]
lr_reg_result2['Standardize'] = True


# In[22]:


results = pd.concat([lr_results, lr_reg_result1, lr_reg_result2])
results = results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
results


# In[23]:


import statsmodels.api as sm
 
lm_fit = sm.OLS(ys, sm.add_constant(xs_onehot)).fit()
print(lm_fit.summary())


# In[24]:


lm_fit.pvalues


# ## Polynomial Regression

# In[25]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

pipe_poly = Pipeline([
    ('poly_transform', PolynomialFeatures()),
    ('standardize', StandardScaler()),
    ('model', Ridge(alpha=10, random_state=42, max_iter=10000))
])

param_grid = {
    'poly_transform__degree': np.arange(1,11,1)
}


# In[26]:


grid_poly = GridSearchCV(pipe_poly, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                         scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_FS, ys)


# In[27]:


poly_result = pd.DataFrame(grid_poly.cv_results_)[['mean_test_score', 'mean_train_score', 'param_poly_transform__degree']]
poly_result


# In[28]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(poly_result['param_poly_transform__degree'], -poly_result['mean_train_score'], linewidth=2, color='r')
plt.xticks(np.arange(1,11))
plt.xlabel('Polynomial Degree k')
plt.ylabel('RMSE')
plt.title('Training')

plt.subplot(1,2,2)
plt.plot(poly_result['param_poly_transform__degree'], -poly_result['mean_test_score'], linewidth=2, color='b')
plt.title('Validation')
plt.xticks(np.arange(1,11))
plt.xlabel('Polynomial Degree k')
plt.ylabel('RMSE')

plt.tight_layout()
plt.show()


# In[29]:


poly_optimal = Ridge(alpha=10, random_state=42, max_iter=1000).fit(PolynomialFeatures(5).fit_transform(xs_top5_FS), ys)


# In[30]:


np.argsort(poly_optimal.coef_)[::-1]


# ## Neural Network

# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# tune the number of neurons for the first hidden layer
pipe_hidden_layer1 = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(random_state=42, max_iter=2000))
])

param_grid = {
    'model__hidden_layer_sizes': [(x,) for x in np.arange(1, 51)]
}


grid_hidden_layer1 = GridSearchCV(pipe_hidden_layer1, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_onehot, ys)


# In[32]:


nn1_results = pd.DataFrame(grid_hidden_layer1.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__hidden_layer_sizes']]
nn1_results = nn1_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
nn1_results.head()


# In[33]:


# tune the number of neurons for the second hidden layer
pipe_hidden_layer2 = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(random_state=42, max_iter=2000))
])

param_grid = {
    'model__hidden_layer_sizes': [(39,x) for x in np.arange(1, 51)]
}


grid_hidden_layer2 = GridSearchCV(pipe_hidden_layer2, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_onehot, ys)


# In[34]:


nn2_results = pd.DataFrame(grid_hidden_layer2.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__hidden_layer_sizes']]
nn2_results = nn2_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
nn2_results.head()


# In[35]:


# tune the regularization term
pipe_reg = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(hidden_layer_sizes=(39,42), random_state=42, max_iter=2000))
])

param_grid = {
    'model__alpha': [10.0**x for x in np.arange(-3,4)]
}


grid_reg = GridSearchCV(pipe_reg, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                        scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_onehot, ys)


# In[36]:


nnreg_results = pd.DataFrame(grid_reg.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__alpha']]
nnreg_results = nnreg_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
nnreg_results


# ## Random Forest

# In[57]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipe_num_features = Pipeline([
    ('standardize', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__max_features': np.arange(0.1,1.1,0.1)
}

grid_num_features = GridSearchCV(pipe_num_features, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                                 scoring='neg_root_mean_squared_error', 
                                 return_train_score=True).fit(xs_onehot, ys)


# In[58]:


rf_features_results = pd.DataFrame(grid_num_features.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__max_features']]
rf_features_results = rf_features_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
rf_features_results.head()


# In[59]:


pipe_num_trees = Pipeline([
    ('standardize', StandardScaler()),
    ('model', RandomForestRegressor(max_features=0.4, random_state=42))
])

param_grid = {
    'model__n_estimators': np.arange(10, 210, 10)
}

grid_num_trees = GridSearchCV(pipe_num_trees, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                                 scoring='neg_root_mean_squared_error', 
                                 return_train_score=True).fit(xs_onehot, ys)


# In[60]:


rf_trees_results = pd.DataFrame(grid_num_trees.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__n_estimators']]
rf_trees_results = rf_trees_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
rf_trees_results


# In[ ]:


pipe_tree_depth = Pipeline([
    ('standardize', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=160, max_features=0.4, random_state=42))
])

param_grid = {
    'model__max_depth': np.arange(1, 31, 1)
}

grid_tree_depth = GridSearchCV(pipe_tree_depth, param_grid=param_grid, cv=10, n_jobs=-1, 
                               verbose=1, scoring='neg_root_mean_squared_error', 
                               return_train_score=True).fit(xs_onehot, ys)


# In[50]:


rf_depth_results = pd.DataFrame(grid_tree_depth.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__max_depth']]
rf_depth_results = rf_depth_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
rf_depth_results.head()


# ## Question 22: 
# Randomly pick a tree in your random forest model (with maximum depth of 4) and plot its structure. Which feature is selected for branching at the root node? What can you infer about the importance of features?

# In[51]:


from sklearn.ensemble import RandomForestRegressor

rf_viz = RandomForestRegressor(n_estimators=160, max_features=0.4, max_depth=4, random_state=42, oob_score=True)
rf_viz.fit(xs_onehot_standard, ys)


# In[52]:


from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image

tree = rf_viz.estimators_[1]
export_graphviz(tree, out_file = 'tree.dot', feature_names = xs_onehot.columns, rounded = True, precision = 1)


# In[53]:


(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
Image('tree.png')


# In[54]:


rf_optim = RandomForestRegressor(n_estimators=160, max_features=0.4, max_depth=22, random_state=42, oob_score=True)
rf_optim.fit(xs_onehot_standard, ys)


# In[55]:


rf_optim.score(xs_onehot_standard, ys)


# In[56]:


rf_optim.oob_score_


# In[ ]:





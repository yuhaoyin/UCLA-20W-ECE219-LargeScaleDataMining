#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

video_df = pd.read_csv('online-video-dataset/transcoding_mesurment.tsv', delimiter='\t')
video_df = video_df.drop(['id', 'b_size', 'umem'], axis=1)


# <font size=4> **Question 1**: Plot a heatmap of Pearson correlation matrix of dataset columns. Report which features have the highest absolute correlation with the target variable and what that implies.</font>

# In[2]:


video_corr = video_df.corr()
video_corr


# In[239]:


import seaborn as sb
import matplotlib.pyplot as plt

plt.figure(figsize=(15,12))
sb.heatmap(video_corr,
            xticklabels=video_corr.columns,
            yticklabels=video_corr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

plt.savefig('heatmap_video.png')


# <font size=4> **Question 2**: Plot the histogram of numerical features. What preprocessing can be done if the distribution of a feature has high skewness? </font>

# In[246]:


xs = video_df.iloc[:, :-1]
ys = video_df.iloc[:, -1]


# In[247]:


xs.columns


# In[248]:


for i in range(xs.shape[1]):
    plt.figure()
    if type(xs.iloc[0,i]) != str:
        plt.hist(xs.iloc[:,i], bins=20, edgecolor='k', facecolor='b', linewidth=1.5, alpha=0.8)
        plt.xlabel(xs.columns[i])
        plt.ylabel('Frequency')
        plt.savefig('histogram{}_video.png'.format(i))


# <font size=4> **Question 3**: Inspect box plot of categorical features vs target variable. What intuition do you get? </font>

# In[244]:


sb.boxplot(xs['codec'], ys, order=list(set(xs['codec'])))
plt.figure()
sb.boxplot(xs['o_codec'], ys, order=list(set(xs['o_codec'])))
plt.show()


# <font size=4> **Question 5**: For video transcoding time dataset, plot the distribution of video transcoding times, what can you observe? Report mean and median transcoding times. </font>

# In[250]:


import numpy as np

sb.distplot(ys, color='b')
plt.savefig('distribution_video.png')
np.mean(ys), np.median(ys)


# <font size=4> **Question 6**: Handling categorical features: *One-hot Encoding*. </font>

# In[8]:


xs_onehot = pd.concat([xs,pd.get_dummies(xs[['codec', 'o_codec']],prefix=['codec', 'o_codec'],drop_first=True)],axis=1).drop(['codec', 'o_codec'],axis=1)


# In[9]:


xs_onehot.head()


# <font size=4> **Question 7**: Standardize feature columns and prepare them for training. </font>

# In[54]:


from sklearn import preprocessing

xs_standard = preprocessing.scale(xs_onehot)


# <font size=4> **Question 8**: You may select most important features using mutual information or F-scores. How does this step affect the performance of your models in terms of test RMSE? </font>

# In[55]:


from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression

MutualInfo = mutual_info_regression(xs_standard, ys)
Fscore = f_regression(xs_standard, ys)


# In[56]:


# select 5 most important variables 

top5_MI = np.argsort(MutualInfo)[::-1][:5]
top5_FS = np.argsort(Fscore[0])[::-1][:5]

xs_top5_MI = xs_onehot.iloc[:, top5_MI]
xs_top5_FS = xs_onehot.iloc[:, top5_FS]


# In[251]:


xs_onehot.columns[top5_FS]


# ## Linear Regression

# <font size=4> **Question 9-11**: Train ordinary least squares, as well as Lasso and Ridge regression, and compare their performances. </font>

# In[79]:


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
xs = [xs_top5_MI, xs_top5_FS]

for x in xs:
    cv_results = cross_validate(pipe_lr_nonstandard, x, ys, scoring='neg_root_mean_squared_error', 
                                return_train_score=True, cv=10)
    lr_train_score.append(np.mean(cv_results['train_score']))
    lr_test_score.append(np.mean(cv_results['test_score']))
    cv_results = cross_validate(pipe_lr_standard, x, ys, scoring='neg_root_mean_squared_error', 
                                return_train_score=True, cv=10)
    lr_train_score.append(np.mean(cv_results['train_score']))
    lr_test_score.append(np.mean(cv_results['test_score']))


# In[80]:


lr_results = pd.DataFrame(data={'mean_test_score': lr_test_score, 'mean_train_score': lr_train_score,
                                'param_model': 'LinearRegression()', 'param_model__alpha': 'NA', 
                                'Standardize': [False, True, False, True], 
                                'Feature Selection': ['Mutual Information', 'Mutual Information', 'F Scores', 'F Scores']})


# In[82]:


lr_results


# In[83]:


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


# In[84]:


grid1 = GridSearchCV(pipe_nonstandard, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_MI, ys)


# In[85]:


grid2 = GridSearchCV(pipe_nonstandard, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_FS, ys)


# In[86]:


grid3 = GridSearchCV(pipe_standard, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_MI, ys)


# In[87]:


grid4 = GridSearchCV(pipe_standard, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_FS, ys)


# In[88]:


lr_reg_result1 = pd.DataFrame(grid1.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model', 'param_model__alpha']]
lr_reg_result1['Standardize'] = False
lr_reg_result1['Feature Selection'] = 'Mutual Information'

lr_reg_result2 = pd.DataFrame(grid2.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model', 'param_model__alpha']]
lr_reg_result2['Standardize'] = False
lr_reg_result2['Feature Selection'] = 'F Scores'

lr_reg_result3 = pd.DataFrame(grid3.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model', 'param_model__alpha']]
lr_reg_result3['Standardize'] = True
lr_reg_result3['Feature Selection'] = 'Mutual Information'

lr_reg_result4 = pd.DataFrame(grid4.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model', 'param_model__alpha']]
lr_reg_result4['Standardize'] = True
lr_reg_result4['Feature Selection'] = 'F Scores'


# In[89]:


results = pd.concat([lr_results, lr_reg_result1, lr_reg_result2, lr_reg_result3, lr_reg_result4])
results = results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
results


# <font size=4> **Question 12**: Some linear regression packages return $p$-values for different features. What is the meaning of them and how can you infer the most significant features? </font>

# In[253]:


import statsmodels.api as sm
 
lm_fit = sm.OLS(ys, sm.add_constant(xs_onehot)).fit()
print(lm_fit.summary())


# In[256]:


lm_fit.pvalues.sort_values(ascending=True)


# ## Polynomial Regression

# <font size=4> **Question 13-14**: Perform polynomial regression by crafting products of raw features up to a certain degree and applying linear regression on the compound features. </font>

# In[106]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

pipe_poly = Pipeline([
    ('poly_transform', PolynomialFeatures()),
    ('standardize', StandardScaler()),
    ('model', Ridge(alpha=100, random_state=42, max_iter=10000))
])

param_grid = {
    'poly_transform__degree': np.arange(1,11,1)
}


# In[107]:


grid_poly = GridSearchCV(pipe_poly, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                         scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_top5_FS, ys)


# In[108]:


poly_result = pd.DataFrame(grid_poly.cv_results_)[['mean_test_score', 'mean_train_score', 'param_poly_transform__degree']]
poly_result


# In[259]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(poly_result['param_poly_transform__degree'], -poly_result['mean_train_score'], linewidth=2, color='r')
plt.xticks(np.arange(1,11))
plt.xlabel('Polynomial Degree k')
plt.ylabel('RMSE')
plt.title('Train')

plt.subplot(1,2,2)
plt.plot(poly_result['param_poly_transform__degree'], -poly_result['mean_test_score'], linewidth=2, color='b')
plt.title('Validation')
plt.xticks(np.arange(1,11))
plt.xlabel('Polynomial Degree k')
plt.ylabel('RMSE')

plt.tight_layout()
plt.show()


# In[129]:


poly_optimal = Ridge(alpha=100, random_state=42, max_iter=1000).fit(PolynomialFeatures(3).fit_transform(xs_standard_top5_FS), ys)


# In[130]:


np.argsort(poly_optimal.coef_)[::-1]


# <font size=4> **Question 15**: For the transcoding dataset it might make sense to craft inverse of certain features
# such that you get features such as $\frac{x_ix_j}{x_k}$, etc. Explain why this might make sense and check if doing so will boost accuracy. </font>

# In[96]:


from sklearn import preprocessing 

x_inverse_feature = np.divide(np.prod(xs_onehot[['o_height', 'o_width']], axis=1), xs_onehot['o_bitrate'], axis=1)
x_inverse_feature = preprocessing.scale(x_inverse_feature)


# In[97]:


xs_inverse_concat = np.concatenate((xs_standard_top5_FS, x_inverse_feature.reshape(-1,1)), axis=1)


# In[98]:


lm_inverse_concat = cross_validate(Ridge(alpha=100, random_state=42, max_iter=1000), 
                                   xs_inverse_concat, ys, scoring='neg_root_mean_squared_error', cv=10)
rmse_inverse_concat = np.mean(-lm_inverse_concat['test_score'])
rmse_inverse_concat


# ## Neural Network

# <font size=4> **Question 17**: Adjust your network size (number of hidden neurons and depth), and weight decay as regularization. Find a good hyper-parameter set systematically. </font>

# In[147]:


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


# In[149]:


nn1_results = pd.DataFrame(grid_hidden_layer1.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__hidden_layer_sizes']]
nn1_results = nn1_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
nn1_results.head()


# In[150]:


# tune the number of neurons for the second hidden layer
pipe_hidden_layer2 = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(random_state=42, max_iter=2000))
])

param_grid = {
    'model__hidden_layer_sizes': [(16,x) for x in np.arange(1, 51)]
}


grid_hidden_layer2 = GridSearchCV(pipe_hidden_layer2, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_onehot, ys)


# In[152]:


nn2_results = pd.DataFrame(grid_hidden_layer2.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__hidden_layer_sizes']]
nn2_results = nn2_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
nn2_results.head()


# In[153]:


# tune the regularization term
pipe_reg = Pipeline([
    ('standardize', StandardScaler()),
    ('model', MLPRegressor(hidden_layer_sizes=(16,40), random_state=42, max_iter=2000))
])

param_grid = {
    'model__alpha': [10.0**x for x in np.arange(-3,4)]
}


grid_reg = GridSearchCV(pipe_reg, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                        scoring='neg_root_mean_squared_error', return_train_score=True).fit(xs_onehot, ys)


# In[154]:


nnreg_results = pd.DataFrame(grid_reg.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__alpha']]
nnreg_results = nnreg_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
nnreg_results


# ## Random Forest

# <font size=4> **Question 20**: Fine-tune your model. Explain how these hyperparameters affect the overall performance? Do some of them have regularization effect?</font>

# In[155]:


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


# In[156]:


rf_features_results = pd.DataFrame(grid_num_features.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__max_features']]
rf_features_results = rf_features_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
rf_features_results.head()


# In[162]:


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


# In[164]:


rf_trees_results = pd.DataFrame(grid_num_trees.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__n_estimators']]
rf_trees_results = rf_trees_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
rf_trees_results


# In[171]:


pipe_tree_depth = Pipeline([
    ('standardize', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=190, max_features=0.4, random_state=42))
])

param_grid = {
    'model__max_depth': np.arange(1, 31, 1)
}

grid_tree_depth = GridSearchCV(pipe_tree_depth, param_grid=param_grid, cv=10, n_jobs=-1, 
                               verbose=1, scoring='neg_root_mean_squared_error', 
                               return_train_score=True).fit(xs_onehot, ys)


# In[175]:


rf_depth_results = pd.DataFrame(grid_tree_depth.cv_results_)[['mean_test_score', 'mean_train_score', 'param_model__max_depth']]
rf_depth_results = rf_depth_results.sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)
rf_depth_results.head()


# <font size=4> **Question 22:** Randomly pick a tree in your random forest model (with maximum depth of 4) and plot its structure. Which feature is selected for branching at the root node? What can you infer about the importance of features? </font>

# In[191]:


rf_viz = RandomForestRegressor(n_estimators=190, max_features=0.4, max_depth=4, random_state=42, oob_score=True)
rf_viz.fit(xs_standard, ys)


# In[217]:


from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image

tree = rf_viz.estimators_[1]
export_graphviz(tree, out_file = 'tree.dot', feature_names = xs_onehot.columns, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
Image(graph.create_png())


# <font size=4> **Question 24:** For random forest model, measure “Out-of-Bag Error” (OOB) as well. Explain what OOB error and R2 score means. </font>

# In[220]:


rf_optim = RandomForestRegressor(n_estimators=190, max_features=0.4, max_depth=26, random_state=42, oob_score=True)
rf_optim.fit(xs_standard, ys)


# In[238]:


print('Optimal Random Forest Regression Model:')
print('R^2 score: %.4f,\nOOB score: %.4f.' %(rf_optim.score(xs_standard, ys), rf_optim.oob_score_))


# In[ ]:





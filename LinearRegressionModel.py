timport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Read and get cursory view of data
customers = pd.read_csv('Ecommerce Customers')  
customers.head()                                
customers.describe()
customers.info()


# Get insight into variables in data
sns.pairplot(customers)            # See which variable(s) correlate best with "Yearly Amount Spent"
                                   # "Length of Membership" and "Time on App" seem like the two variable with strong positive correlations to "Yearly Amount Spent"

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers,kind='scatter')
sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=customers,kind='scatter') # "Length of Membership" is the stronger predictor
sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=customers,kind='hex')


# Training and Testing Data
customers.columns        # View all columns from data set

X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']] # Use all numerical factors as features, exceprt for "Yearly Amount Spent"
Y = customers['Yearly Amount Spent']    # Find predictions for "Yearly Amount Spent"

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)


# Training Data
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
lm.coef_      #Print out coefficients from model


# Predicting Test Data
predictions = lm.predict(X_test)
plt.scatter(Y_test,predictions)     # Compare test data to predictions


# Evaluating the Model
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
# Results: MAE: 7.228148653430853
#          MSE: 79.81305165097487
#          RMSE: 8.933815066978656

sns.distplot((Y_test-predictions))  # Observe residual errors

# Looking at coefficients, length of membership  is the best predictor for how much the member spends on clothing from this company.
# If they want to improve business, they should focus more on improving their website, since their website has not been contributing much to sales.

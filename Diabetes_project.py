from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt



#In this part, the data is being loaded.
from sklearn.linear_model import LinearRegression

def diabetes_model():

    diabetes = load_diabetes()
    print(type(diabetes))


    print(diabetes.DESCR)

    #Here, a Pandas DataFrame is created.
    df = pd.DataFrame(diabetes.data)

    df.columns = diabetes.feature_names
    print(df.head())

    df['people_age'] = diabetes.target
    print(df.head())

    x = df.drop('people_age',axis=1)#We don't want the ages as a dependent variable.
    y = df.age

    #Creating a linear regression model.
    l = LinearRegression()

    print(l.fit(x,y))
    print("Predict:\n", l.predict(x))

    plt.scatter(df.people_age,l.predict(x))
    plt.xlabel('Original Ages')
    plt.ylabel('Predicted ages.')
    plt.title('Original Ages vs Predicted Ages.')
    plt.show()

diabetes_model()







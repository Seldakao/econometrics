# How is DiD approach different from CUPED (using pre treatment data as a regressor)

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# load data

df = pd.read_stata('data/Panel101.dta')

unit_identifier = 'country'
date_identifier = 'year'

treatment_group = ['E', 'F', 'G']
intervention_time = 1994
outcome = 'y'
regressors = ['treated', 'time', 'did']

def print_something(df):
    """print something about a dataframe all in one go """
    print(df.info())
    print(df.head())
    print(df[unit_identifier].unique())
    print(df[date_identifier].unique())

def did(df=df):
    """
    Difference in Difference analysis with an example
    :return:
    """
    #print_something(df)


    df = df.set_index(df[date_identifier])

    # creating a treated variable to identify target (1) and the holdout (0) groups
    df['treated'] = np.where(df['country'] >= 'E', 1, 0)
    # Adding time variable, treatment occured in 1994 ; segmenting anything before 1994 as pre-period (0)
    df['time'] = np.where(df.index >= 1994, 1, 0)
    df['did'] = df['treated'] * df['time']

    aggregate = df.groupby([df.index, 'treated']).mean()
    treated_outcome = aggregate.loc[aggregate.index.isin([1], level=1), outcome].reset_index()
    untreated_outcome = aggregate.loc[aggregate.index.isin([0], level=1), outcome].reset_index()

    # Diff and Diff Estimator (beta 3)
    x = df[regressors]
    y = df[outcome]
    x2 = sm.add_constant(x)
    est = sm.OLS(y, x2)
    est2 = est.fit()
    print(est2.summary())

    # plot time series of a treated and untreated over time
    plt.plot(treated_outcome.year, treated_outcome.y)
    plt.plot(untreated_outcome.year, untreated_outcome.y)
    plt.legend(['treated', 'untreated'])
    #plt.show()


def cuped(df=df):
    "get estimator using pretreatment data as one of the regressor"

    # creating a treated variable to identify target (1) and the holdout (0) groups
    df['treated'] = np.where(df['country'] >= 'E', 1, 0)
    # Adding time variable, treatment occured in 1994 ; segmenting anything before 1994 as pre-period (0)
    df['time'] = np.where(df[date_identifier]>= 1994, 1, 0)

    df = df.set_index(df[unit_identifier]).drop(columns=unit_identifier)

    aggregate = df.groupby([df.index, 'time']).mean()
    pre_data = aggregate.loc[aggregate.index.isin([0], level=1), :].reset_index().set_index(unit_identifier).sort_index()
    post_data = aggregate.loc[aggregate.index.isin([1], level=1),:].reset_index(unit_identifier).set_index(unit_identifier).sort_index()

    merged_data = pre_data.merge(post_data, left_index=True, right_index = True, how = 'inner', suffixes=['_pre', '_post'])
    print(merged_data.columns)
    y = merged_data['y_post']
    x = merged_data[['treated_pre', 'y_pre']]
    x2 = sm.add_constant(x)
    est = sm.OLS(y, x2)
    est2 = est.fit()
    print(est2.summary())




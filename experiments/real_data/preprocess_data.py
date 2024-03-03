import pandas as pd
import numpy as np
from scipy.io import arff


def preprocess_compas(compas_filepath):
    '''
    https://github.com/propublica/compas-analysis and https://github.com/dylan-slack/Modeling-Uncertainty-Local-Explainability/blob/main/bayes/data_routines.py
    '''
    selected_features = ['age', 'two_year_recid','c_charge_degree', 'race', 'sex', 'priors_count',
                         'days_b_screening_arrest', 'c_jail_out', 'c_jail_in', 'score_text']
    compas_df = pd.read_csv(compas_filepath, usecols=selected_features)
    compas_df = compas_df.loc[(np.abs(compas_df['days_b_screening_arrest']) <= 30)].reset_index(drop=True)
    compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) -
                                   pd.to_datetime(compas_df['c_jail_in'])).dt.days
    compas_df['score'] = compas_df['score_text'].map({'High': 0, 'Medium': 1, 'Low': 1})
    compas_df['sex'] = compas_df['sex'].map({'Male': 1, 'Female': 0})
    compas_df['c_charge_degree'] = compas_df['c_charge_degree'].map({'M': 1, 'F': 0})
    compas_df = compas_df.drop(['days_b_screening_arrest', 'c_jail_out', 'c_jail_in', 'score_text'], axis=1)
    sens = compas_df.pop('race')
    sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
    compas_df['race'] = sensitive_attr
    assert all((sens == 'African-American') == (compas_df['race'] == 1))
    compas_df.columns = compas_df.columns.str.title()
    return compas_df, 'Score'


def preprocess_bike(bike_filepath):
    '''
    Bike sharing data from Kaggle competition.
    Located at: https://www.kaggle.com/c/bike-sharing-demand
    From SAGE https://github.com/iancovert/sage/blob/master/notebooks/bike.ipynb
    '''
    bike_df = pd.read_csv(bike_filepath)

    # Split and remove datetime column.
    bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])
    bike_df.insert(loc=0, column='year', value=bike_df['datetime'].dt.year)
    bike_df.insert(loc=1, column='month', value=bike_df['datetime'].dt.month)
    bike_df.insert(loc=2, column='day', value=bike_df['datetime'].dt.day)
    bike_df.insert(loc=3, column='hour', value=bike_df['datetime'].dt.hour)
    bike_df = bike_df.drop(['datetime', 'casual', 'registered'], axis=1)
    bike_df.columns = bike_df.columns.str.title()
    return bike_df, 'Count'


def preprocess_nomao(nomao_filepath):
    '''
    Nomao data from OpenML.
    Located at: https://www.openml.org/search?type=data&sort=runs&id=1486&status=active
    '''
    data, _ = arff.loadarff(nomao_filepath)
    nomao_df = pd.DataFrame(data)
    nomao_df.columns = nomao_df.columns.str.title()
    ordinal_features = ['V7', 'V8', 'V15', 'V16', 'V23', 'V24', 'V31', 'V32', 'V39', 'V40', 'V47',
                            'V48', 'V55', 'V56', 'V63', 'V64', 'V71', 'V72', 'V79', 'V80', 'V87', 'V88',
                            'V92', 'V96', 'V100', 'V104', 'V108', 'V112', 'V116', 'Class']
    nomao_df[ordinal_features] = nomao_df[ordinal_features].apply(lambda x: x.str.decode('utf8'))
    nomao_df[ordinal_features] = nomao_df[ordinal_features].astype('int')
    nomao_df['Class'] = nomao_df['Class'].map({2: 1, 1: 0})
    return nomao_df, 'Class'


def get_data_by_name(name, filepath):
    supported_data = {'compas': preprocess_compas,
                      'bike': preprocess_bike,
                      'nomao': preprocess_nomao}
    if not name in supported_data.keys():
        raise NameError('Supported filenames are: %s', ', '.join(supported_data.keys()))
    return supported_data[name](filepath)










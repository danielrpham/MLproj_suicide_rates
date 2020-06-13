import cleaning as clg
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sns.set()


def age_groups(data):
    # just takes in the suicide data first
    # use suicides/100k pop and sum by age
    # take note of any countries with nulls or inequal counts?

    # grouping the ages first
    # data_age = data.groupby(['age', 'year'])['suicides/100k pop'].sum()
    data_age = data.groupby(['age', 'year']).\
                    agg({'suicides/100k pop': ['sum']}).reset_index()
    data_age.columns = ['age', 'year', 'sum']
    # grouping by the country
    sns.relplot(x='year', y='sum', kind='line', hue='age', data=data_age)
    plt.xlabel('Year')
    plt.ylabel('Countries sum of suicides/100k pop')
    plt.title('Suicide Rates by Age Group')
    plt.xticks(list(range(1985, 2017, 3)))
    plt.xticks(rotation=60)
    plt.savefig('/home/age_groups.png', bbox_inches='tight')
    data_country = data.groupby(['year', 'country']).\
        agg({'suicides/100k pop': ['sum']}).reset_index()
    data_country.columns = ['year', 'country', 'sum']
    # plot the 10 highest  #
    data_max = data_country.groupby(['country']).\
        agg({'sum': ['max']}).reset_index()
    data_max.columns = ['country', 'max']
    max_list = data_max.nlargest(10, 'max')['country']
    data_country_max = data_country[data_country['country'].isin(max_list)]
    # cant really do it with min because the lowest won't be that interesting
    # lots of zeros

    sns.relplot(x='year', y='sum', kind='line',
                hue='country', data=data_country_max)
    plt.xlabel('Year')
    plt.ylabel('Countries sum of suicides/100k pop')
    plt.title('Suicide Rates by Country with Maximums')
    plt.xticks(list(range(1985, 2017, 3)))
    plt.xticks(rotation=45)
    plt.savefig('/home/max_country_groups.png', bbox_inches='tight')


def plot_shapes(shape_data, centroid_data):
    """
    The purpose of this method is to first plot the shape of the countries
    and then plot the centroids that are generated for the country.
    This is important to know for a visual understanding of how the
    latitude of the centroids will be used
    """
    fig, ax = plt.subplots(1, figsize=(15, 7))

    shape_data.plot(ax=ax)
    centroid_data.plot(color='red', markersize=10, ax=ax)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Countries shapes with centroids')
    plt.savefig('/home/countries_shapes_and_centroids.png')


def model_predicting_latitude(data):
    """
    this method will take in the suicide rates joined
    with the geometry of the country and then see if
    latitudes are a good predictor for the suicide rates.
    ## note that the countries rates are NOT summed
    ## for over the years
    ## single linear regression model
    """
    data_country = data.groupby(['year', 'country']).\
        agg({'suicides/100k pop': ['sum']}).reset_index()
    data_country.columns = ['year', 'country', 'sum']
    data = data[['NAME_EN', 'lat']]
    data = data.drop_duplicates(subset='NAME_EN')
    data = data.dropna(subset=['NAME_EN', 'lat'])

    merged = data_country.merge(data, left_on='country', right_on='NAME_EN',
                                how='inner')
    merged = merged.drop(columns=['NAME_EN'])

    print('----merged header---\n', merged.head(10))
    print('\n' * 10)
    # first let's plot the data to see linearity
    sns.relplot(x='lat', y='sum', data=merged)
    plt.savefig('/home/first_glance_lat.png', bbox_inches='tight')
    # would want to look at the abs value of lat
    # because it's stored as negative too
    merged.loc[:, 'lat'] = merged['lat'].abs()
    sns.relplot(x='lat', y='sum', data=merged)
    plt.savefig('/home/abs_glance_lat.png', bbox_inches='tight')
    plt.clf()
    # doesn't look super appropriate but can just run the test
    X = merged['lat'].values.reshape(-1, 1)
    y = merged['sum'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print('regressor, intercept: ', regressor.intercept_)
    print('regressor, slope: ', regressor.coef_)
    y_pred = regressor.predict(X_test)
    plt.scatter(X_test, y_test,  color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=4)
    plt.xlabel('latitude (absolute)')
    plt.ylabel('Countries sum of suicides/100k pop')
    plt.title('Plotted Lat (x) by Summed Suicide Rates (y)'
              'with linear regression')
    plt.savefig('/home/lat_linear_regression.png')
    print('coefficient of determination = ', regressor.score(X_train, y_train))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('lat sum data mean:', merged['sum'].mean())


def education_test(data):
    """
    this method will take in the suicide rates joined
    with the educaiton rates and then see if it is a
    good predictor for the suicide rates. Note that
    this is only for ages 15 and above.
    ## using multiple linear regression
    ## variables edu and year and HDI
    """
    data = data[data['age'] != '5-14 years']
    data2 = \
        data.groupby(['year', 'country', '% Tertiary Edu', 'HDI for year'])\
            .agg({'suicides/100k pop': ['sum']}).reset_index()
    data2.columns = list(map(''.join, data2.columns.values))

    data2 = data2.dropna(subset=['year', '% Tertiary Edu', 'HDI for year'])
    X = data2[['year', '% Tertiary Edu', 'HDI for year']]
    y = data2['suicides/100k popsum']

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, X.columns,
                            columns=['Coefficient'])
    print(coeff_df)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)
    print('coefficient of determination', regressor.score(X_train, y_train))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Actual Data mean: ', data2['suicides/100k popsum'].mean())


def main():
    sns.set()
    countries_df = gpd.read_file('/course/lecture-readings/geo_data/'
                                 'ne_110m_admin_0_countries.shp')
    centroid_df = clg.clean_countries(countries_df)
    education_df = pd.read_csv('/home/share-of-the-population-with-com'
                               'pleted-tertiary-education.csv')
    suicide_rates_df = pd.read_csv('master.csv')
    suicide_rates_and_shapes = \
        clg.merge_suicide_rates_and_shapes(suicide_rates_df, centroid_df)
    suicide_rates_and_edu = \
        clg.merge_suicide_rates_and_edu(suicide_rates_df, education_df)
    age_groups(suicide_rates_df)
    model_predicting_latitude(suicide_rates_and_shapes)
    plot_shapes(countries_df, centroid_df)
    education_test(suicide_rates_and_edu)


if __name__ == '__main__':
    main()

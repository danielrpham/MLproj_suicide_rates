import pandas as pd
import geopandas as gpd


def clean_countries(countries_df):
    """
    This method takes in the countries dataframe and then
    cleans it. It also adds the centroid, latitude, and longitude
    columns.
    """
    countries_df = countries_df[['NAME_EN', 'geometry']]
    countries_df.loc[:, 'centroid'] = countries_df.loc[:].centroid
    countries_df.loc[:, 'lat'] = countries_df.loc[:].centroid.x
    countries_df.loc[:, 'long'] = countries_df.loc[:].centroid.y
    countries_df = gpd.GeoDataFrame(countries_df, geometry='centroid')
    return countries_df


def merge_suicide_rates_and_shapes(suicide_rates_df, countries_df):
    """
    This method takes in the suicide rate dataframe and the
    countries shape dataframe and joins them.
    """
    merged = suicide_rates_df.merge(countries_df, left_on='country',
                                    right_on='NAME_EN', how='left')
    return merged


def merge_suicide_rates_and_edu(suicide_rates_df, education_df):
    """
    This method takes in the suicide rates dataframe and the
    education dataframe and joins them by the country and year
    """
    merged = pd.merge(suicide_rates_df, education_df,  how='inner',
                      left_on=['country', 'year'],
                      right_on=['suicsuicssfdsEntity', 'Year'])
    merged.columns = ['country', 'year', 'sex', 'age', 'suicides_no',
                      'population', 'suicides/100k pop', 'country-year',
                      'HDI for year', 'gdp_for_year ($)', 'gdp_per_capita ($)',
                      'generation', 'suicsuicssfdsEntity', 'Code', 'Year',
                      '% Tertiary Edu']
    return merged

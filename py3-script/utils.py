""" 
	utils.py: Collection of python functions used in FinHack
"""

# Load modules 
import pandas as pd


def load_excel_df(path_datafile):
	"""
	   Loads a excel data file into Pandas DataFrame  
	"""
	return pd.read_excel(path_datafile, sheet_name=None)


def drop_duplicate_rows(df):
	"""
	   Checks for duplicate rows/instances and drops the rows from dataframe (inplace)
	"""

	# No. of duplicated rows
	ndup_rows = get_duplicate_rows(df)

	print('There are {} duplicated rows in the dataset.'.format(ndup_rows))
	if (ndup_rows > 0):
		return df.drop_duplicates().reset_index(inplace=True, drop=True)
		print('Dropped {} rows from the dataset.'.format(ndup_rows))



def get_duplicate_rows(df):
	"""
	   Returns duplicate rows/instances in the dataframe
	"""
	return df.duplicated().sum()


def get_unique_values(df, colname):
	"""
		Returns a list with all unique values in the column of datafram df
	"""
	return list(dict(df[colname].value_counts(ascending=False, dropna=False)).keys())


def get_unique_counts(df, colname):
	"""
		Returns a list with all counts of unique values in the column of datafram df
	"""
	return list(dict(df[colname].value_counts(ascending=False, dropna=False)).values())


def show_features_datatypes(df):
	"""
	   Prints a table of Features and their DataTypes
	"""
	for inum,icol in enumerate(df.columns):
		print('Column id: {0:3d} \tName: {1:12s} \tDataType: {2}'.format(inum, icol, df[icol].dtypes))


def show_feature_summary(df, colname, display_uniques=False):
	"""
		Prints all necessary information to fix missing data
	"""
	print('Details of feature:',colname)
	print('         - datatype:',df[colname].dtypes)
	print('         - col.size:',df[colname].shape)
	print('         - NaN.vals:',df[colname].isnull().sum())
	if (display_uniques): print('         - uniqvals:',get_unique_values(df, colname))
	if (display_uniques): print('         - cnt.vals:',get_unique_counts(df, colname))
	print("\n")





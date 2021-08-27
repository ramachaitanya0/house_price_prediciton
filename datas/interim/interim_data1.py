# importing libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# reading the data from csv file
df  = pd.read_csv('/Users/karanam.rama/Downloads/house_price_pred_repo/datas/raw/house_pred_raw_data.csv')

# changing the data type of mssubclass
df['MSSubClass'] =df['MSSubClass'].astype('str')

# replacing null values 
df['Fence']=df['Fence'].fillna('NA')
df['PoolQC'] = df['PoolQC'].fillna('NA')
df['MiscFeature'] = df['MiscFeature'].fillna('NA')
df['Alley'] = df['Alley'].fillna('NA')
df['Fence'] = df['Fence'].fillna('NA')
df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
df['LotFrontage'] = df['LotFrontage'].fillna(0)


ndf = pd.DataFrame(df.isnull().mean())
ndf.columns = ['null_mean']
ndf = ndf[ndf['null_mean']>0]
ndf['column_name'] = ndf.index


num_list = df[ndf['column_name']].select_dtypes(include=['number']).columns
obj_list = df[ndf['column_name']].select_dtypes(exclude=['number']).columns

# replacing null of int type columns with their mean values
df[num_list] = SimpleImputer(strategy='mean').fit_transform(df[num_list])

# replacing null of object type columns with the most frequent category
df[obj_list] = SimpleImputer(strategy='most_frequent').fit_transform(df[obj_list])

df.set_index(['Id'],inplace = True)

x = df.drop('SalePrice',axis = 1)
y = df['SalePrice']
all_columns = x.columns
cat_columns = x.select_dtypes(include=['object']).columns
num_columns = x.select_dtypes(exclude=['object']).columns

# creating a new data frame with numerical columns only
num_x = x[num_columns]

# creating a new data frame with categorical columns only
ohe = OneHotEncoder()
data = ohe.fit_transform(x[cat_columns])
cat_x =  pd.DataFrame(data.todense(),columns =ohe.get_feature_names(cat_columns),index = x.index )
cat_x.head()

# joining the both dataframes 

final_x = num_x.join(cat_x,how='inner')

final_data = final_x
final_data['sale_price'] = y

final_data.to_csv('/Users/karanam.rama/Downloads/house_price_pred_repo/datas/interim/interm_dataset1.csv')
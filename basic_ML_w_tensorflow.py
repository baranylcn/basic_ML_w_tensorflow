from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import tensorflow as tf
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.simplefilter(action='ignore', category=Warning)

# Load dataset.
df = pd.read_csv("Customer-Churn-Records.csv") # Dataset
df.columns = df.columns.str.replace(' ', '_')

df.head()
"""
   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age  Tenure   Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited  Complain  Satisfaction_Score Card_Type  Point_Earned
0          1    15634602  Hargrave          619    France  Female   42       2      0.00              1          1               1        101348.88       1         1                   2   DIAMOND           464
1          2    15647311      Hill          608     Spain  Female   41       1  83807.86              1          0               1        112542.58       0         1                   3   DIAMOND           456
2          3    15619304      Onio          502    France  Female   42       8 159660.80              3          1               0        113931.57       1         1                   3   DIAMOND           377
3          4    15701354      Boni          699    France  Female   39       1      0.00              2          0               0         93826.63       0         0                   5      GOLD           350
4          5    15737888  Mitchell          850     Spain  Female   43       2 125510.82              1          1               1         79084.10       0         0                   5      GOLD           425
"""

df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
# Dropped "RowNumber," "CustomerId," and "Surname" columns as they are not suitable variables for model building.

cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "Complain",
            "Card_Type", "Satisfaction_Score", "NumOfProducts"]
num_cols = [col for col in df.columns if col not in cat_cols + ["Exited"]]
# Specified categorical and numerical variables.

def check_dataframe(dataframe, n=5):
    print("*** HEAD ***")
    print(dataframe.head(n))
    print("*** TAIL ***")
    print(dataframe.tail(n))
    print("*** SHAPE ***")
    print(dataframe.shape)
    print("*** DESCRIBE ***")
    print(dataframe.describe().T)
    print("*** NULL VALUES ***")
    print(dataframe.isna().sum())
check_dataframe(df)

def cat_summary(df, cat_cols):
    for col in cat_cols:
        print(df[col].value_counts(dropna=False))

cat_summary(df, cat_cols)
"""
Geography
France     5014
Germany    2509
Spain      2477
Name: count, dtype: int64
Gender
Male      5457
Female    4543
Name: count, dtype: int64
HasCrCard
1    7055
0    2945
Name: count, dtype: int64
IsActiveMember
1    5151
0    4849
Name: count, dtype: int64
Complain
0    7956
1    2044
Name: count, dtype: int64
Card_Type
DIAMOND     2507
GOLD        2502
SILVER      2496
PLATINUM    2495
Name: count, dtype: int64
Satisfaction_Score
3    2042
2    2014
4    2008
5    2004
1    1932
Name: count, dtype: int64
NumOfProducts
1    5084
2    4590
3     266
4      60
Name: count, dtype: int64
"""

# Label encoding for model building:
label_encoder = LabelEncoder()
le_cols = ["Geography", "Gender", "Card_Type"]
for col in le_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Scaling for model building:
scale_cols = ["CreditScore","Balance","EstimatedSalary","Point_Earned"]
rs = RobustScaler()
df[scale_cols] = rs.fit_transform(df[scale_cols])


y = df["Exited"]
X = df.drop(["Exited"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Creating feature columns for TensorFlow model.
feature_columns = []
# Handling categorical columns with vocabulary lists.
for feature_name in cat_cols:
  vocabulary = X[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

# Handling numerical columns as numeric features.
for feature_name in num_cols:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# Define a function for creating input pipelines for TensorFlow model training and evaluation.
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        # Create a TensorFlow Dataset from input data and labels.
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        # Shuffle the dataset if specified.
        if shuffle:
            ds = ds.shuffle(1000)
        # Batch the dataset and repeat for the specified number of epochs.
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function
# Create input functions for training and evaluation datasets.
train_input_fn = make_input_fn(X_train, y_train)
eval_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)


# Modeling.
model = tf.estimator.LinearClassifier(feature_columns=feature_columns)
model.train(train_input_fn)

result = model.evaluate(eval_input_fn)
# Result:
"""
'accuracy': 0.999,
'accuracy_baseline': 0.8,
'auc': 0.9999898,
'auc_precision_recall': 0.9999595,
'average_loss': 0.008777343,
'label/mean': 0.2,
'loss': 0.008780617,
'precision': 0.9975,
'prediction/mean': 0.20005378,
'recall': 0.9975,
'global_step': 2500
"""

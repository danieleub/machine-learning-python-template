from utils import db_connect
engine = db_connect()

# your code here


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
data.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)
data["room_type"] = pd.factorize(data["room_type"])[0]
data["neighbourhood_group"] = pd.factorize(data["neighbourhood_group"])[0]
data["neighbourhood"] = pd.factorize(data["neighbourhood"])[0]

#outliers de precio
price_stats = data["price"].describe()
price_iqr = price_stats["75%"] - price_stats["25%"]
upper_limit = price_stats["75%"] + 1.5 * price_iqr
lower_limit = price_stats["25%"] - 1.5 * price_iqr
data = data[data["price"] > 0]
count_0 = data[data["price"] == 0].shape[0]
count_1 = data[data["price"] == 1].shape[0]


#outliers de precio
nights_stats = data["minimum_nights"].describe()
nights_iqr = nights_stats["75%"] - nights_stats["25%"]
upper_limit = nights_stats["75%"] + 1.5 * nights_iqr
lower_limit = nights_stats["25%"] - 1.5 * nights_iqr
data = data[data["minimum_nights"] <= 15]
count_0 = data[data["minimum_nights"] == 0].shape[0]
count_1 = data[data["minimum_nights"] == 1].shape[0]
count_2 = data[data["minimum_nights"] == 2].shape[0]
count_3 = data[data["minimum_nights"] == 3].shape[0]
count_4 = data[data["minimum_nights"] == 4].shape[0]

#outliers numb_reviews
review_stats = data["number_of_reviews"].describe()
review_iqr = review_stats["75%"] - review_stats["25%"]
upper_limit = review_stats["75%"] + 1.5 * review_iqr
lower_limit = review_stats["25%"] - 1.5 * review_iqr


# calculated_host_listings_count
hostlist_stats = data["calculated_host_listings_count"].describe()
hostlist_iqr = hostlist_stats["75%"] - hostlist_stats["25%"]
upper_limit = hostlist_stats["75%"] + 1.5 * hostlist_iqr
lower_limit = hostlist_stats["25%"] - 1.5 * hostlist_iqr
count_0 = sum(1 for x in data["calculated_host_listings_count"] if x in range(0, 5))
count_1 = data[data["calculated_host_listings_count"] == 1].shape[0]
count_2 = data[data["calculated_host_listings_count"] == 2].shape[0]
total_data = data[data["calculated_host_listings_count"] > 4]

total_data.isnull().sum().sort_values(ascending = False)

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(total_data[num_variables])
df_scal = pd.DataFrame(scal_features, index = total_data.index, columns = num_variables)
df_scal["price"] = total_data["price"]

X = df_scal.drop("price", axis = 1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()
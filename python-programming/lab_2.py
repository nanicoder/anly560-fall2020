# 1.Import pandas under the name pd.
import pandas as pd

# 2.Print the version of pandas that has been imported.
pd.__version__

# 3.Print out all the version information of the libraries that are required by the pandas library.
pd.show_versions()

# 4.Create a DataFrame df from this dictionary data which has the index labels.
import numpy as np

data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data, index=labels)
print(df)

# 5.Display a summary of the basic information about this DataFrame and its data.
df.info()

# 6.Return the first 3 rows of the DataFrame df.
df.head(n=3)

# 7.Select just the 'animal' and 'age' columns from the DataFrame df.
print(df[['animal', 'age']])

# 8.Select the data in rows [3, 4, 8] and in columns ['animal', 'age'].
print(df.loc[df.index[[3, 4, 8]], ['animal', 'age']])

# 9.Select only the rows where the number of visits is greater than 3.
print(df[df['visits'] > 3])

# 10.Select the rows where the age is missing, i.e. is NaN.
print(df[df['age'].isnull()])

# 11.Select the rows where the animal is a cat and the age is less than 3.
print(df[(df['animal'] == 'cat') & (df['age'] < 3)])

# 12.Select the rows the age is between 2 and 4 (inclusive).
print(df[(df['age'] >= 2) & (df['age'] <= 4)])

# 13.Change the age in row 'f' to 1.5.
df.loc['f', 'age'] = 1.5
print(df)

# 14.Calculate the sum of all visits (the total number of visits).
print(df['visits'].sum())

# 15.Calculate the mean age for each different animal in df.
print(df.groupby('animal')['age'].mean())

# 16.Append a new row 'k' to df with your choice of values for each column. Then delete that row to return the original DataFrame.
## add row k:
df.loc['k'] = ['a', 'b', 'true', 'false']
print(df)
## delete row k:
df = df.drop('k')
print(df)
# 17.Count the number of each type of animal in df.
print(df['animal'].value_counts())

# 18.Sort df first by the values in the 'age' in decending order, then by the value in the 'visits' column in ascending order.
# (so row i should be first, and row d should be last).
df.sort_values(by=['age', 'visits'], ascending=[False, True])

# 19.The 'priority' column contains the values 'yes' and 'no'.
# Replace this column with a column of boolean values: 'yes' should be True and 'no' should be False.
df['priority'] = df['priority'].map({'yes': True, 'no': False})

# 20.In the 'animal' column, change the 'snake' entries to 'python'.
df['animal'] = df['animal'].replace('snake', 'python')
print(df)

# 21.For each animal type and each number of visits, find the mean age. In other words,
# each row is an animal, each column is a number of visits and the values are the mean ages (hint: use a pivot table).
# df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean')

# 22.How do you filter out rows which contain the same integer as the row immediately above?
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
df.drop_duplicates(subset='A')

# 23.How do you subtract the row mean from each element in the row?
df = pd.DataFrame(np.random.random(size=(5, 3)))
df.sub(df.mean(axis=1), axis=0)

# 24.Which column of numbers has the smallest sum? (Find that column's label.)
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
df.sum().idxmin()

# 25.How do you count how many unique rows a DataFrame has (i.e. ignore all rows that are duplicates)?
df = pd.DataFrame(np.random.randint(0, 2, size=(10, 3)))
len(df.drop_duplicates(keep=False))

# 26. You have a DataFrame that consists of 10 columns of floating--point numbers.
# Suppose that exactly 5 entries in each row are NaN values. For each row of the DataFrame,
# find the column which contains the third NaN value.
nan = np.nan
data = [[0.04, nan, nan, 0.25, nan, 0.43, 0.71, 0.51, nan, nan],
        [nan, nan, nan, 0.04, 0.76, nan, nan, 0.67, 0.76, 0.16],
        [nan, nan, 0.5, nan, 0.31, 0.4, nan, nan, 0.24, 0.01],
        [0.49, nan, nan, 0.62, 0.73, 0.26, 0.85, nan, nan, nan],
        [nan, nan, 0.41, nan, 0.05, nan, 0.61, nan, 0.48, 0.68]]

columns = list('abcdefghij')
df = pd.DataFrame(data, columns=columns)
(df.isnull().cumsum(axis=1) == 3).idxmax(axis=1)

# 27.For each group, find the sum of the three greatest values.
df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'),
                   'vals': [12, 345, 3, 1, 45, 14, 4, 52, 54, 23, 235, 21, 57, 3, 87]})
df.groupby('grps')['vals'].nlargest(3).sum(level=0)
print(df)

# 28 The DataFrame df constructed below has two integer columns 'A' and 'B'. The values in 'A' are between 1 and 100 (inclusive).
# For each group of 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...), calculate the sum of the corresponding values in column 'B'.
# The answer should be a Series as follows:
df = pd.DataFrame(np.random.RandomState(8765).randint(1, 101, size=(100, 2)), columns=["A", "B"])

# write a solution to the question here
df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()

# 29 Consider a DataFrame df where there is an integer column 'X':
df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
# For each value, count the difference back to the previous zero (or the start of the Series, whichever is closer). These values should therefore be
# [1, 2, 0, 1, 2, 3, 4, 0, 1, 2]
# Make this a new column 'Y'.
x = (df['X'] != 0).cumsum()
y = x != x.shift()
df['Y'] = y.groupby((y != y.shift()).cumsum()).cumsum()

# 30.Consider a DataFrame containing rows and columns of purely numerical data.
# Create a list of the row-column index locations of the 3 largest values.
df = pd.DataFrame(np.random.RandomState(30).randint(1, 101, size=(8, 8)))
df.unstack().sort_values()[-3:].index.tolist()

# 31.Given a DataFrame with a column of group IDs, 'grps', and a column of corresponding integer values, 'vals',
# Create a new column 'patched_values' which contains the same values as the 'vals'
# any negative values in 'vals' with the group mean:
df = pd.DataFrame({"vals": np.random.RandomState(31).randint(-30, 30, size=15),
                   "grps": np.random.RandomState(31).choice(["A", "B"], 15)})


def replace(x):
    negative_value = x < 0
    x[negative_value] = x[~negative_value].mean()
    return x


df['patched_vals'] = df.groupby('grps')['vals'].transform(replace)
df

# 32.Implement a rolling mean over groups with window size 3, which ignores NaN value. For example consider the
# following DataFrame:
df = pd.DataFrame({'group': list('aabbabbbabab'),
                   'value': [1, 2, 3, np.nan, 2, 3, np.nan, 1, 7, 3, np.nan, 8]})

# group values
g = df.groupby(['group'])['value']
# compute means
s = g.rolling(3, min_periods=1).mean()
# drop/sort index
s.reset_index(level=0, drop=True).sort_index()

# 33.Create a DatetimeIndex that contains each business day of 2015 and use it to index a Series of random numbers.
# Let's call this Series s.
datetime = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B')
s = pd.Series(np.random.rand(len(datetime)), index=datetime)
print(s)

# 34.Find the sum of the values in s for every Wednesday.
s[s.index.weekday == 2].sum()

# 35.For each calendar month in s, find the mean of values.
s.resample('M').mean()

# 36.For each group of four consecutive calendar months in s, find the date on which the highest value occurred.
s.groupby(pd.Grouper(freq='4M')).idxmax()

# 37.Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016.
pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')

# 38.Some values in the the FlightNumber column are missing. These numbers are meant to increase by 10 with each
# row so 10055 and 10075 need to be put in place. Fill in these missing numbers and make the column an integer column (instead of a float column).
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
                               'Budapest_PaRis', 'Brussels_londOn'],
                   'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
                   'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
                               '12. Air France', '"Swiss Air"']})
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)

# 39.The From_To column would be better as two separate columns! Split each string on the underscore delimiter _ to give a new temporary DataFrame with the correct values. Assign the correct column names to this temporary DataFrame
TempDF = df.From_To.str.split('_', expand=True)
TempDF.columns = ['From', 'To']

# 40.Notice how the capitalisation of the city names is all mixed up in this temporary DataFrame.
# Standardise the strings so that only the first letter is uppercase (e.g. "londON" should become "London".)
TempDF['From'] = TempDF['From'].str.capitalize()
TempDF['To'] = TempDF['To'].str.capitalize()

# 41.Delete the From_To column from df and attach the temporary DataFrame from the previous questions.
df = df.drop('From_To', axis=1)
df = df.join(TempDF)

# 42.In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names.
# Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.
df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()

# 43.In the RecentDelays column, the values have been entered into the DataFrame as a list.
# We would like each first value in its own column, each second value in its own column, and so on.
# If there isn't an Nth value, the value should be NaN.
#    Expand the Series of lists into a DataFrame named delays, rename the columns delay_1, delay_2, etc.
#    and replace the unwanted RecentDelays column in df with delays.
delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns) + 1)]
df = df.drop('RecentDelays', axis=1).join(delays)

# 44.Given the lists letters = ['A', 'B', 'C'] and numbers = list(range(10)), construct a MultiIndex object from the product of the two lists.
# Use it to index a Series of random numbers. Call this Series s.
letters = ['A', 'B', 'C']
numbers = list(range(10))
multi_index = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=multi_index)

# 45.Check the index of s is lexicographically sorted (this is a necessary proprty for indexing to work correctly with a MultiIndex).
s.index.is_lexsorted()

# 46.Select the labels 1, 3 and 6 from the second level of the MultiIndexed Series.
s.loc[:, [1, 3, 6]]

# 47.Slice the Series s; slice up to label 'B' for the first level and from label 5 onwards for the second level.
s.loc[pd.IndexSlice[:'B', 5:]]

# 48.Sum the values in s for each label in the first level (you should have Series giving you a total for labels A, B and C).
s.sum(level=0)

# 49.Suppose that sum() (and other methods) did not accept a level keyword argument. How else could you perform the equivalent of s.sum(level=1)?
s.unstack().sum(axis=0)

# 50.Exchange the levels of the MultiIndex so we have an index of the form (letters, numbers). Is this new Series properly lexsorted? If not, sort it.
new_s = s.swaplevel(0, 1)
new_s.index.is_lexsorted()
# the result is "False", therefore sort it.
new_s = new_s.sort_index()

# 56. Pandas is highly integrated with the plotting library matplotlib, and makes plotting DataFrames very user-friendly!
# Plotting in a notebook environment usually makes use of the following boilerplate:
import matplotlib.pyplot as plt

# matplotlib inline
plt.style.use('ggplot')
df = pd.DataFrame({"xs": [1, 5, 2, 8, 1], "ys": [4, 2, 1, 9, 6]})

# 57. Columns in your DataFrame can also be used to modify colors and sizes.
# Bill has been keeping track of his performance at work over time, as well as how good he was feeling that day,
# and whether he had a cup of coffee in the morning. Make a plot which incorporates all four features of this DataFrame.
df = pd.DataFrame({"productivity": [5, 2, 3, 1, 4, 5, 6, 7, 8, 3, 4, 8, 9],
                   "hours_in": [1, 9, 6, 5, 3, 9, 2, 9, 1, 7, 4, 2, 2],
                   "happiness": [2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 2, 1, 3],
                   "caffienated": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0]})

df.plot.scatter("hours_in", "productivity", s=df.happiness * 30, c=df.caffienated)

# 58. What if we want to plot multiple things? Pandas allows you to pass in a matplotlib Axis object for plots,
# and plots will also return an Axis object.
# Make a bar plot of monthly revenue with a line plot of monthly advertising spending (numbers in millions)
df = pd.DataFrame({"revenue": [57, 68, 63, 71, 72, 90, 80, 62, 59, 51, 47, 52],
                   "advertising": [2.1, 1.9, 2.7, 3.0, 3.6, 3.2, 2.7, 2.4, 1.8, 1.6, 1.3, 1.9],
                   "month": range(12)
                   })
ax = df.plot.bar("month", "revenue", color="blue")
df.plot.line("month", "advertising", secondary_y=True, ax=ax)
ax.set_xlim((-1, 12))

import numpy as np


def float_to_time(x):
    return str(int(x)) + ":" + str(int(x % 1 * 60)).zfill(2) + ":" + str(int(x * 60 % 1 * 60)).zfill(2)


def day_stock_data():
    # NYSE is open from 9:30 to 4:00
    time = 9.5
    price = 100
    results = [(float_to_time(time), price)]
    while time < 16:
        elapsed = np.random.exponential(.001)
        time += elapsed
        if time > 16:
            break
        price_diff = np.random.uniform(.999, 1.001)
        price *= price_diff
        results.append((float_to_time(time), price))

    df = pd.DataFrame(results, columns=['time', 'price'])
    df.time = pd.to_datetime(df.time)
    return df


# Don't read me unless you get stuck!
def plot_candlestick(agg):
    """
    agg is a DataFrame which has a DatetimeIndex and five columns: ["open","high","low","close","color"]
    """
    fig, ax = plt.subplots()
    for time in agg.index:
        ax.plot([time.hour] * 2, agg.loc[time, ["high", "low"]].values, color="black")
        ax.plot([time.hour] * 2, agg.loc[time, ["open", "close"]].values, color=agg.loc[time, "color"], linewidth=10)

    ax.set_xlim((8, 16))
    ax.set_ylabel("Price")
    ax.set_xlabel("Hour")
    ax.set_title("OHLC of Stock Value During Trading Day")
    plt.show()


# 59 Generate a day's worth of random stock data,
# and aggregate / reformat it so that it has hourly summaries of the opening, highest, lowest, and closing prices

df = day_stock_data()
df.head()

df.set_index("time", inplace=True)
agg = df.resample("H").ohlc()
agg.columns = agg.columns.droplevel()
agg["color"] = (agg.close > agg.open).map({True: "green", False: "red"})
agg.head()

# 60 Now that you have your properly-formatted data, try to plot it yourself as a candlestick chart.
# Use the plot_candlestick(df) function above, or matplotlib's plot documentation if you get stuck.o
plot_candlestick(agg)

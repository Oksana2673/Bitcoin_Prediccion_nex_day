# Bitcoin_Prediccion_nex_day
In this notebook we will try to predict, using all the data related to Bitcoin of the day, the closing price of the next day. To do this, we will use the `SGDRegressor` algorithm from `sklearn` with the *batches* training method from `rolling forecast windows`.

First, we will use the `.Ticker()` method of the `yfinance` library to be able to invoke the economic data related to bitcoin. Later we will use the `.history()` method to receive the data via API of the Bitcoin price history. For this project, a historical period of ten years in an interval of consecutive days has been used, a priori.

The dataframe columns represent the following:
- **Open**: Open price of Bitcoin at the beginning of the day. Unit of measurement: United States dollars (USD).
- **High**: Highest price (High) is the maximum price reached by Bitcoin during the corresponding day. Unit of measurement: United States dollars (USD).
- **Low**: Lowest price (Low) is the minimum price reached by Bitcoin during the corresponding day. Unit of measurement: United States dollars (USD).
- **Close**: Closing price (Close) is the price of Bitcoin at the end of the day. Unit of measurement: United States dollars (USD).
- **Volume**: Total amount of Bitcoin traded during the specified time period. Unit of measurement: Bitcoin (BTC) units.
- **Dividends**: Dividends (Dividends) represent the payments made to the shareholders of a company as part of the profits generated.
- **Stock Splits**: Stock Splits are events in which the number of stock units outstanding is adjusted, generally increasing the number of units available without changing the total market value.

- As we see, the **Dividends** and **Stock Splits** columns only contain null values. This is logical, since Bitcoin, being a cryptocurrency, neither performs splits nor distributes dividends. Therefore, we will eliminate those columns.

- Then we will create some synthetic variables to reinforce the information contained in the dataframe.

- Once this is done, we are going to proceed to create the target variable of our predictive model. As indicated at the beginning, this project consists of predicting the price (Close) of the next day using, to do so, the data of the current day. For this purpose, we are going to create a new column in our dataframe that, in each row, will contain the price (Close) of the next day, as indicated in the following scheme:
- <img src="https://images2.imgbox.com/32/3b/qdEZMBxc_o.jpg" alt="drawing" width="400"/>

Once this is done, all rows, except the last one, will have the Close price of that day and that of the next.

The price (Close) of the day will remain as the main marker when predicting the price (Close) of the next day.

A very important issue is that, once the model has been trained with this scaled data, we must use exactly the same scaler (without redefining it) if we want to add new data without retraining the model.

Having said this, we proceed with the training loop. As we said at the beginning, we are going to use the "rolling windows" method; That is, instead of feeding the model with all the points, we will gradually give it temporary windows with a fixed size and displacement. A training outline is as follows:

<img src="https://images2.imgbox.com/71/ef/X91uYzbu_o.jpg" alt="drawing" width="400"/>

Another important issue will be the use of the `.partial_fit()` method in training our regression model. According to the `SGDRegressor()` documentation, the use of `.fit()` is recommended for general training; that is, when the entire dataset is used as training data (except those reserved for **test**, of course). For this reason, `.fit()`, when applied a second time, discards all previous information, preventing the system from retraining using partial data. All this is solved with the `.partial_fit()` method, which is used to retrain the model with new data as it becomes available. It is recommended for **rolling forecast** type training; that is, those in which the data arrives temporarily.


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

![image](https://github.com/Oksana2673/Bitcoin_Prediccion_nex_day/assets/127301922/109efa11-ea8e-4809-878f-4cf6610961c2)

What we see in the previous graphs is that, as our model learns, the error gradually decreases. The notable one is the sharp rise, again, of the error in time window 18. This is due to the sharp (and unexpected) rise in the price of Bitcoin at the end of 2020 and beginning of 2021. After this, we see that the model stabilizes and begins to adjust properly to the target variable.

In order to be able to check, graphically, the difference between the prediction and the actual price values, we are going to enter the predictions into our dataframe. Logically, due to the way of training, the first 400 values lack a prediction; Therefore, we fill these with NaN values:

After this, we can see what the structure of our dataframe looks like. In this way, we have "price_next_day" as the real price and, to its right, "price_next_day_pred" as the prediction made by the model in the successive tests.

![image](https://github.com/Oksana2673/Bitcoin_Prediccion_nex_day/assets/127301922/2dd29a37-4d11-442b-b62e-5ff5fd965e3b)

Once this is done, all that remains is to make the prediction for the next day for today and, hopefully, hope that it is right to make us rich :)

### Extra: Prediction using a neural network

A relatively standard architecture within "time series forecasting" has been used as the neural network architecture. This consists of 4 layers:
- An LSTM layer with 100 network units. Set `return_sequence` to `True` so that the output of the layer is another sequence of the same length.
- Another LSTM layer, also 100 network units. But this time, set `return_sequence` to `False` so that it only returns the last output in the sequence.
- A densely connected neural network layer with 30 network units.
- A densely connected layer that specifies the output of 1 network unit, corresponding to the next day's price prediction.

- As an optimizer, we will use `Adam` as it is another of the standards for this type of problems. Adjusted `learning_rate` to its default value of 0.001; however, this has been done explicitly so that it can be changed more easily in testing. The least squares error is used as the loss function.

- From here, the same training principle with time windows has been carried out using the neural network. Early stopping has been configured to prematurely stop training if the value of the loss function begins to increase during training.

Similarly, 20% of the total training values have been used as a validation set. The parameter `shufle=False` has been set at all times, so that the order of the data is not altered.

Let's see how our model has behaved in terms of errors in the different training time windows:

![image](https://github.com/Oksana2673/Bitcoin_Prediccion_nex_day/assets/127301922/c1e7de14-2f1c-4d64-8a90-818dc8402e6a)

We add, as we did with the predictions of the linear regression model, the values predicted by the neural network to our bitcoin price dataframe:

![image](https://github.com/Oksana2673/Bitcoin_Prediccion_nex_day/assets/127301922/c9b9dfe7-734b-41da-86c3-bbbf3b512fa3)





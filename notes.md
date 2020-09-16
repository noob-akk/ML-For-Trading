


# Lesson 2: Reading and plotting stock data


## Stock Header explaination
 
* **Open**: opening price of the stock  
* **High**: HIghest price of the stock during the day  
* **Low**: Lowest price of the stock during the day  
* **Close**: Closing price of the stock at the end of the day  
* **Volume**: Total number of stocks traded during the day  
* **Adj Close**: Adjusted close is the adjusted value of close for dividend splits etc  


## Pandas Dataframe

* **NaN** is not a number or does not exist  
* Pandas can also handle df's in a 3D-way
* `pd.read_csv()` reads the csv file
* `pd.head(n)` returns the first n rows of df (default = 5)
* Similarly, `pd.tail(n)` returns the last n rows
* To get data from rows n1 to n2, use `df[n1,n2+1]`
* Mean. Max, Min are callable functions like `df['Volume'].max()`


# Lesson 3: Working with multiple stocks


## Problems to solve

* Date Ranges
* Multiple stocks
* Align dates
* Proper Date order


## Building DataFrame

* `join` you join the columns with intersection of indexes
* `parse_dates=True` will parse trhe dates into date objects
* `usecols` param can be used to define what colums to keep in the dataset
* `index_col` param can be used for defining the index column in df
* `Pandas.DataFrame.dropna()` will drop all the rows with `NaN` value
* `Pandas.DataFrame.rename()` to rename a specific column in the dataframe
* `Pandas.DataFrame.dropna(subset=[])` : subset can be used to direct which columsn to be checked for na and then to be dropped


## Slicing DataFrame

* `df.loc`, `df.iloc` can be used for row splicing;loc for label based splicing, iloc is position based splicing
* column splicing can be done by passing a list of columns inside `df[[x,y]]`


### Normalize the data to compare multiple stocks


# Lesson 4: The power of Numpy


### Intro

* `np.empty(shape)` to generate empty array;But the array retains the memory values
* `np.ones(shape)` to create ones' array
* `np.random.random(shape)` to create uniformly random values in $[0,1)$
* `np.random.normal(mean=0, std=1, shape)` for normal distribution
* `np.random.randint(min, max, shape)` to generate random integer array


### Array attributes

* `array.shape` = shape tuple 
* `array.size` = number of elems
* `array.dtype`


### Operations on ndarrays

* `axis=0` for operations on that axis; `a.sum(axis=0)` would give sum of all the rows by maintaining the columns
* `a.min(axis)`, `a.max(axis)`
* `a.mean()`
* `np.argmax()`
* `a[i,j]`
* `a[i,m:n:step]`
* You can pass array of indices to sample as `a[idxs]` where `idx` is the array of indices you want to sample 
* boolean masks can be put on 2D arrrays and extract the elements under the mask by `a[condition]`
* $*$ operator does element by element not matrix multiplication and similarly $/$ does elem division


# Lesson 5: Statistical analysis of time series


### Global Statistics

* `df.mean()`, `df.median()`, `df.std()`


### Rolling Statistics

* rolling statistics important argument of TA
* __Bollinger bands__ are drawn $2*rolling\\_std$ to identify buy and sell signals

* __Daily Returns__ is $curr\\_day\\_price/prev\\_day\\_price - 1$


# Lesson 6: Incomplete Data


### Reality of stock data

* Data is an amalgamation from multiple sources
* Not all stocks trade at all time
* Gaps in data are common


### Why data goes missing

* Companies who trade once in a while generate gaps
* Also companies when acquired by a new company etc also begin or end abruptly


### What to do?

* Fill forward from the latest data in the entire gap
* At the beginning of stock, you can fill backward in time
* `fillna(method=)` can be used to fill;`method=ffill` for forward, `method=bfill` for backward
* in the `fillna` function, argument `inplace` can be set to true to modify inplace


# Lesson 7: Histograms and Scatter plots

__Histogram of daily retuns__ looks like a bell curve i.e., gaussian distribution
* Mean, Std can be measured from the histogram
* Kurtosis can be measured
  * Fat tails means positive kurtosis
  * Skinny tails means negative kurtosis
* Mean less => Lower return
* Std less => Lower Volatility


## $\\alpha$ & $\\beta$ definitions

* $\\alpha$, $\\beta$ are the intercept, slope of the line fit between market and stock
* can be used to check relative performance of the stock compared to market


### Correlation $not = \\beta$
* can be computed by `df.corr(method='pearson')`
* correlation is how tightly the data fits the line


# Lesson 8: Sharpe ratio & other portfolio stats


### Daily portfolio calue estimation
* normalize prices
* multiply with allocations of each stock (which sum to 1 altogether
* mutliply by initial investment
* `df.sum(axis=1)` will give value fo portfolio each day



### Portfolio Stats
* Daily returns
  * exclude the initial zero of day 1
  * `daily_ret = daily_ret[1:]`
* cumulative return
  * `(port_val[-1]/port_val[0])-1`
* average return
  * `daily_ret.mean()`
* standard dev of daily returns
  * `daily_ret.std()`
* __sharpe_ratio__


### Sharpe Ratio
* by william sharpe
* measures risk adjusted return
    * lower risk ($\\sigma_p$) is better
    * higher return ($R_p$) is better
* volatility is considered as risk
* SR also considers risk free rate of return($R_f$) (like interest rate of banks)
* Sharpe Ratio$~=~\\frac{E[R_p-R_f]}{std[R_p-R_f]}$
* if $R_f$ is constant, which it usually is, it vanishes from the denominator


### More on sharpe ratio
* SR varies based on how we sample
* SR is an snnualized measure
* $SR_{annualized}~=~\\sqrt{Samples~per~year}*SR$
* If we are sampling daily, $SR_{annualized}~=~\\sqrt{252}*SR$


# Lesson 9: Optimizers: Building a parametrized model


### Example of optimizers in scipy

__Convex function__ is  a real-valued function f(x) defined on an interval is called convex if the line segment between any two points on the graph of the function lies above the graph


# Lesson 10: Optimizers: How to optimize a portfolio


### Framing the problem
* Optimize the allocations of the portfolio in order to maximize sharpe ratio 
* loss function = $-1*SR$


### Ranges and Allocations
* Limit the search range
    * Ex: put in zero to 1
* Constraints
    * Sum of the allocations to be 1


# Lesson 11: So you want to be a hedge fund manager


### Types of funds

* __Exchange Traded Funds__
    * Has 3/4 letters
    * Buy/Sell like stocks
    * Baskets of stocks
    * Transparent and are very liquid (easy to trade)

* __Mutual Funds__
    * Usually has 5 letters
    * Buy/Sell at EOD
    * Quaterly disclosure
    * Less Transparent

* __Hedge Funds__
    * Buy/Sell by agreement (secret)
    * Hard to exit a hedge fund
    * No disclosure about what they are holding
    * Not transparent

__AUM__:  *Assets Under Management* is the total amount of money being managed by the fund.


### How are these fund managers compensated?

* ETFs
    * Expense ratio 0.01% to 1.00% (fraction of AUM)
* Mutual Funds
    * Expense ratio 0.5% to 3%
* Hedge Funds
    * Two and Twenty
        * 2% of AUM and 20% of profit
    * Now-a-days they charge a little less
    * A very few charge even more


### How hedge funds attract investors?

* Who?
    * Individuals
        * Wealthy folks
    * Institutions
        * Large retirement foundations
        * University foundations
    * Funds of funds
        * grouping of money
        
* Why?
    * Track record
    * Simulation/Back test our strategy
    * Good portfolio fit for the investors


### Hedge fund goals and metrics

* __Goals__
    * Beat a benchmark
        * Ex: Beat the SP500 index
    * Absolute return
        * Positive return no matter what
            * __Long__ Positive bets in funds going up 
            * __Short__ Negative bets in funds going down
            
* __Metrics__
    * Cumulative return
    * Volatility
        * Standard deviation
    * Risk/Reward
        * Sharpe Ratio


# Lesson 12: Market Mechanics


### What's an order?

* Buy or Sell?
* Symbol of the stock
* #shares
* Limit Order or Market Order
    * Limit is playing safe
    * Market order no price spec required
* Examples
    * BUY, IBM, 100, LIMIT, 99.95
    * SELl, GOOG, 150, MARKET


### The order book

* Keeps track of sell, buy orders
* Allots the buyers to sellers of their price req match
* If no match, appends into the book


### Example of order book in operation

| __BID/ASK__ | __PRICE__ | __SIZE__ |
|:---------:|:-------:|:------:|
|ASK|100.10|100|
|ASK|100.05|500|
|ASK|100.00|1000|
|BID|99.95|100|
|BID|99.90|50|
|BID|99.85|50|
[Book at the beginning]

1. Order Buy,100, Market
    * 100 shares from 1000 shares cut & **Price = 100$**


| __BID/ASK__ | __PRICE__ | __SIZE__ |
|:---------:|:-------:|:------:|
|ASK|100.10|100|
|ASK|100.05|500|
|ASK|100.00|__900__|
|BID|99.95|100|
|BID|99.90|50|
|BID|99.85|50|
[Book after order 1]


2. Order Buy, 100, Limit, 100.02
    * 100 shares from 900 are deducted & **Price = 100 $**
  
  
| __BID/ASK__ | __PRICE__ | __SIZE__ |
|:---------:|:-------:|:------:|
|ASK|100.10|100|
|ASK|100.05|500|
|ASK|100.00|__800__|
|BID|99.95|100|
|BID|99.90|50|
|BID|99.85|50|
[Book after order 2]

3. Order Sell, 175, Market
    * 100, 50, 25 will be deducted from the three bids respectively
    * Executed price is average of these 175, **99.66$**


| __BID/ASK__ | __PRICE__ | __SIZE__ |
|:---------:|:-------:|:------:|
|ASK|100.10|100|
|ASK|100.05|500|
|ASK|100.00|800|
|BID|99.95|**0**|
|BID|99.90|**0**|
|BID|99.85|**25**|
[Book after order 3]

4. Because of the selling pressure being higher than buying pressure, the prices will likely go down


### How Hedge funds explot market mechanics

* HF will buy and sell you the stock by the time the order you placed reached stock exchange because of the delay in 10-20 ms
* **Geographic arbitrage** HF have multiple locations like NY, London. If the same stock has different prices in these places, HF will buy in the lower price city and sell in higher price city. They made profits ez


### Additional order types

* Exchanges
    * Buy and Sell
    * Market and Limit
    
* Broker
    * Stop Loss
    * Stop Gain
    * Trailing Stop
    * **Sell Short**
    

### Mechanics of short selling

* Sell without owning the stock yet
* Sell at higher price, buy&exit at lower price
* Hence you make profit


### What can go wrong in short selling?

* If prices go after short selling, you make loss when you exit the position


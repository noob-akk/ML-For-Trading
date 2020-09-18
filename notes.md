


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


# Lesson 13: What is a company's worth?

* True value of company
* Stock price goes above true value, sell
* Stock price goes below true value, buy


### Ways to estimate worth

**Intrinsic Value**  
Value of the company estimated by dividends

**Book value**  
Value of the assets the company owns  
Ex: Factories etc

**Market Cap**  
Value of the stock and number of outstanding stocks  


### The value of a future dollar
Present Value$~=~\\frac{Future~Value}{(1+Interest~Rate)^{Years}}$

**Intrinsic Value** $=~\\sum\\frac{Future~Value}{(1+Interest~Rate)^{i}}$  

$~~~~~~~~~~~~~~~~~~~~~~= \\frac{Future~Value*(1+Interest~Rate)}{Interest~Rate}$  

$~~~~~~~~~~~~~~~~~~~~~~= \\frac{Future~Value}{Discount~Rate}$

**Book Value** = Total value of the assets minus intangible assets and liabilities

**Market Capitalization** = #shares * price


# Lesson 14: Capital Assets Pricing Model (CAPM)


### Portfolio

* $r_{s}$, $w_{s}$ are returns, weights of each stock in portfolio
* Return of portfolio $r_{p}~=~\\sum r_{s}*w_{s}$
* $\\sum abs(w_{s})~=~1$


### Market Portfolio

* Every country has index of stocks
* Weight of each stock in the index is $w_{i}~=~Market~Cap_{i}/\\sum Market~Cap_{j}$
* SP500 is a market portfolio of 500 stocks


### CAPM Equation
$r_{i}(t)=\\beta_{i}*r_{m}(t)+\\alpha_{i}(t)$
* Return of markets: $r_{m}(t)$
* Return of stock i: $r_{i}(t)$
* Significant portion of return of the stock is from the growth of market
* Most stocks have $\\beta$ as 1
* CAPM says $E[\\alpha_{i}(t)]=0$; Its essentially a random variable
* $\\beta,~\\alpha$ are **slope, intercept** of line fitted between stock, market daily returns' scatter


### CAPM vs Active management

* passive: buy index portfolio and hold
* active: pick stocks with different weights compared to index
    * Some have underweight
    * Some have overweight
* CAPM says $E[\\alpha_{i}(t)]=0$; $\\alpha_{i}(t)$ is fully random
* Active managers believe we can predict alpha and its not fully random


### CAPM for portfolios

* Return of portfolio $r_{p}~=~\\sum w_{i}*(\\beta_{i}*r_{m}(t)+\\alpha_{i}(t))$
* Beta of the portfolio is $\\beta_{p}=\\sum w_{i}*\\beta_{i}$
* $r_{p}(t)=\\beta_{p}*r_{m}(t)+\\sum w_{i}*\\alpha_{i}(t)$
    * CAPM would say alpha term is simply a random number with mean zero $\\alpha_{p}(t)$
    * Active managers would use the summation formulation


### Arbitrage Pricing Theory

* Beta of market can be broken down into betas of different fields like fincance, tech, manufacturing etc


# Lesson 15: How Hedge funds use the CAPM


### Two stock CAPM math

* $r_p = (w_a \\beta_a + w_b \\beta_b)*r_m + w_a \\alpha_a + w_b \\alpha_b$
* If we make net beta as zero, we equate to zero market risk in case of long short investing


### Summary

* Assuming 
    * We forecasted alpha
    * We got beta pf the stock
    
* CAPM enables
    * minimize market risk by making beta of portfolio to be zero
    * allocate weighst to stocks ion the portfolio such that above happens 


# Lesson 16: Technical Analysis


### Characteristics of TA
* Uses price and volume only
* Compute indicators by statistics on data
* There is information in price


### When is TA effective
* Individual indicators are weak
* Combining multiple indicators makes predictions stronger
* Look for contrasts (Stock vs Market)
* Works better for shorter time periods than longer time periods
* Fundamental factors have higher value when horizon is longer


### Momentum

* n-day momentum, $momentum_{t} = \\frac{price_{t}}{price_{t-n}}-1$
* -0.5 to +0.5


### Simple moving average

* n-day SMA, IndicatorSMA$_t = \\frac{price_{t}}{\\frac{\\sum price_t}{n}}-1$
* -0.5 to +0.5
* Could be used as proxy for underlying value especially with window is large
    * If price above the average, it might indicate the stock is overpriced and bound to fall
    * If stock is below average, it might indicate the stock is undervalued and bound to rise
* The points where moving average curve and price curve intesects could be used as trading signals



### Bollinger Bands

* have moving average and moving standrd deviation curves drawn
    * When price is going to fall inside the band at +2$\\sigma$, it could be a sell signal
    * When price is goign to rise inside the band at -2$\\sigma$, it could be a buy signal
* -1 to +1


### Normalization

* Technical indicators ahve to normalized to allot equal importance to each indicators


# Lesson 17: Dealing with Data

* data is sampled at lowest resolution called tick which indicates a successful transaction (only when buy and sell match)

* **Stock Split** happens usually when the stock becomes very high price (n:1)
    * creates issue when we feed in data to our algorithm
    * adjusted close will adjust the prices by back tracking
        * At the latest day, actual price and adjusted close match

* **Dividends** comapny payouts
    * Adjusted close will reduce the price on the day previous to dividend pay by an amount of dividend
        * At the latest day, actual price and adjusted close match
        

### Survivorship Bias

* Use survivor bias free data


# Lesson 18: Efficient Market Hypothesis


### Assumptions

* Large number of investors
* New info arrives randomly
* prices adjust quickly
* Proces reflect all available information


### Three forms of EMH

* **Weak** Future prices cannot be predicted by historical prices
* **Semi-Strong** Prices adjust rapidly to new public info
* **Strong** Prices reflect all information public and private


### Not considered to be true


# Lesson 19: The Fundamental Law of active portfolio management


### Information Ratio (IR)

* $return_p = \\beta_p *market + \\alpha_p$
* $\\alpha_p$ is the skill of fund manager of portfolio
* IR = $\\frac{mean(\\alpha_p)}{std(\\alpha_p)}$
* Sharpe ratio of excess return


### Information Coefficient (IC)

* Correlation of forecasts to returns
* 0 to 1


### Breadth (BR)

* Number of trading oppurtunities per year


### Grinold's Fundamental Law

* $performance=skill*\\sqrt{breadth}$
* $IR=IC*\\sqrt{BR}$ 


## Coin flipping casino example

* 1000 tables are tossing coins
* You have 1000 tokens to bet
* coins are biased with head's probability being 0.51


### Option 1: Betting all tokens on a single table

* Expected return = $1000*0.51-1000*0.49 = 20$
* But the risk is too high
    * standard deviation = $\\sqrt{1000^2 - 20^2}=1000$
* risk discounted reward = $\\frac{20}{1000}=0.02$


### Option 2: Betting a token each on all tables

* Expected return = $(0.51-0.49)*1000=20$
* Standard deviation = $\\sqrt{1000*(0.51+0.49)1^2 - 20^2}=\\sqrt{1000-400}=24.49$
* risk discounted reward = $\\frac{20}{24.49}=0.82$


### Observations

* SR_2 = 0.82 >> SR_1 = 0.02
* SR_2 ~= SR_1* $\\sqrt{1000}$
* Coincidence? I think not.


# Lesson 20: Portfolio optimization and the efficient frontier

$*$ **Risk is $\\sigma_{daily returns}$**


### Why covariance matters?

* Combining anti-correlated stocks would reduce risk i.e., standard deviation
* Combining strongly correlated stocks wouldn't add much to return or risk of the portfolio which actually might remain nearly the same


### Mean Variance Optimization (MVO)

* Inputs:
    * Expected return
    * volatility
    * covariance of daily returns
    * target return of the portfolio
* Output
    * Asset weights for portfolio that minimizes risk
    


### The Efficient Frontier

![efficient-frontier.png](attachment:efficient-frontier.png)

* Dont go on the curve to a point where you reward is less for the same amount of risk
* If you are inside the curve, we can do better
* Tangent to the curve gives max sharpe ratio
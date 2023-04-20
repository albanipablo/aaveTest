# AAVE ML Yield Strategy

This pool has 4 jupyter notebooks that were run sequentially.

The data is not provided due to its size (+4 GB), but the results are left in the corresponding cells.

Each file consists of the following process:

- **1.aave-web3-data-gathering**:

Download the log data corresponding to the AAVE contract, directly from a node of each network (more reliable than TheGraph), for each network and version of the protocol that interests us.
In addition, we add a timestamp to each event from the block to be able to relate the different networks



- **2.aave-preproc**:

The data from the logs is then pre-processed, combining the fund movement events (Supply, Withdraw, Borrow, Repay, FlashLoan) with the rate modification events in the contract (ReserveDataUpdated).

The data types are converted to numeric and additional metrics such as the accumulated deposited and borrowed are also calculated.

Then, the data is grouped in a time interval of 5 minutes, which will be the granularity with which the model will be trained. For that, the variables are recalculated as sums, averages or first value of the interval.

The Dataset to predict the "liquidityRate" (rate charged for providing liquidity) of the USDT token in the V2 protocol of ETH Mainet is built.

Finally, the data of the other pools, protocols and networks are stored with the same dimensionality and calculations as this first base.



- **3.other-data-sources**:

To help the predictive power of the model, other external information is added to the AAVE protocol that is supposed to help improve the prediction. They are:

**Binance**: Price and volume information on Binance (the largest current CEX) for the same time granularity (5 min) and for the relevant token pairs on AAVE

**CoinGecko**: Information aggregated by CoinGecko for relevant tokens, including market sentiment, internal scores, circulating supply, and Reddit metrics.

**Twitter**: Tweets from a Bigquery table, where data from Twitter referring to +1000 Crypto accounts from 2017 onwards are stored, we create variables for the same temporal dimensionality (5 min) from where we get:

The number of tweets with that mention, responsesretweets & likes



- **4.aave-modeling**:

In the final notebook, a small Feature Engineering is first done by combining all the variables from the previously mentioned datasets and the resulting variables are analyzed.

Then we proceed to train a LightGBM model with 10 CV folds

And finally, an optimization of Hyperparameters is carried out on the basis of training / validation and it is retested on the basis of testing.

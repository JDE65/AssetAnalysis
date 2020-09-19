# AssetAnalysis - WIP
Asset analysis : downloading Assets infos and correlated assets for analysis
The schema always follows the structure presented herewith.

The repository includes (and/or will include) various codes to :
0. Load the libraries
1. Ask for inputs : hyper-parameters & parameters
2. Connect to DBs and load the data
 - download stock data (date, close, open, high, volume, daily traded volume, ...)
 - dowload financial data of "correlated assets" (major indices, ST and LT interest rates, commodities, ...)
 - enrich data with technical indicators (MA & EMA, MACD, Bollinger band, CCI, EMV, ATR, ADX, RSI, MOM)
 - enrich data with price from assets of the same class
 - enrich data with correlated assets (FX, rate, commodities, macro-economic data, ...)
 - transform series for detailed analysis (Fourier, ARIMA, ...)
3. Organise the data into matrixes X and y
  - Create the Dataset
  - Enrich the Dataset
  - Organize matrix X
  - Organize matrix y
  - Clean X & y for invalid data
  - Normalize X
  - Separate into train & test sets
4. Organize the model architecture
  - Dense NN
  - LSTM NN
  - Conv1D NN
  - ConvLSTM 1D NN
  - Bayesian LSTM
  -...
5. Compile + Train + Test the model
6. Perform scenario analysis for these  NN
 - Compute the expected results of the strategy
 - Analyze number of deals, IRR, average investment, Sharpe & Sortino ratios
 - Plot the marked-to-market 
 - Plot average daily return
 
 ... TBC

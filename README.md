# AssetAnalysis - OLD
Asset analysis : downloading Assets infos and correlated assets for analysis

The repository includes (and/or will include) the codes to :
1. Prepare the data
 - download stock data from Refinitiv (date, close, open, high, volume, daily traded volume, ...)
 - dowload financial data of "correlated assets" (major indices, ST and LT interest rates, commodities, ...)
 - enrich data with technical indicators (MA & EMA, MACD, Bollinger band, CCI, EMV, ATR, ADX, RSI, MOM)
 - enrich data with price from assets of the same class
 - enrich data with correlated assets
 - transform series for detailed analysis (Fourier, ARIMA)
 - 
 2. Organise the data into matrixes X and y
  - Create the Dataset
  - Enrich the Dataset
  - Organize matrix X
  - Organize matrix y
  - Clean X & y for invalid data
  - Normalize X
  - Separate into train & test sets
  
 3. Run "simple" neural networks
  - Dense NN
  - LSTM NN
  - Conv1D NN
  - ConvLSTM 1D NN
  
 4. Perform scenario analysis for these "simple" NN
 
 5. Run Conv2D NN on multi-assets model (multi-to-multi)
 
 6. Run Bayesian Dense NN
 ... TBC

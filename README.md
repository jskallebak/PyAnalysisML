# PyAnalysisML

A Python library for cryptocurrency OHLC data analysis, feature engineering with pandas-ta, and price prediction using machine learning.

## Features

- Fetch historical OHLC data from Binance
  - Support for parallel data fetching for improved performance with large datasets
  - Customizable chunking for optimal memory usage and API rate limits
- Calculate technical indicators using pandas-ta
- Engineer custom features for improved prediction
- Train and evaluate machine learning models
- Visualize results and predictions

## Installation

### Prerequisites

- Python 3.8 or higher

### Package Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pyanalysisml.git
cd pyanalysisml

# Install the package in development mode
pip install -e .
```

## Usage

```python
from pyanalysisml.data.binance_client import BinanceClient
from pyanalysisml.features.technical_indicators import add_indicators
from pyanalysisml.models.model_factory import create_model
from pyanalysisml.models.train import train_model
from pyanalysisml.utils.visualization import plot_predictions

# Initialize Binance client and fetch data
client = BinanceClient()
df = client.get_historical_klines("BTCUSDT", "1d", "1 year ago")

# Add technical indicators
df = add_indicators(df)

# Train model
X, y, X_test, y_test = prepare_data(df)
model = create_model("random_forest")
model = train_model(model, X, y)

# Evaluate and visualize
predictions = model.predict(X_test)
plot_predictions(df.iloc[-len(y_test):], y_test, predictions)
```

### Parallel Data Fetching

For large datasets, you can utilize parallel processing to speed up data retrieval:

```python
from pyanalysisml.data.data_loader import load_from_binance
import time

# Load data with parallel processing
start_time = time.time()
df = load_from_binance(
    symbol="BTCUSDT",
    interval="1d",
    start_str="2 years ago",
    chunk_size=30,  # Split into 30-day chunks
    max_workers=4   # Use 4 workers in parallel
)
print(f"Parallel loading took {time.time() - start_time:.2f} seconds")
```

Check the `examples/` directory for more detailed usage examples.

## Technical Indicators

This project uses `pandas-ta` for calculating technical indicators, which provides a large collection of technical analysis indicators implemented in pure Python. Some of the indicators used include:

- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Average True Range (ATR)
- Stochastic Oscillator
- Average Directional Index (ADX)
- On Balance Volume (OBV)
- Rate of Change (ROC)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

import yfinance as yf
import pandas as pd

# Define the ticker symbol
ticker_symbol = 'SAVE'

# Load data for each and every day from January 1, 2000, to the current date
stock_data = yf.download(ticker_symbol, start='2000-01-01', end='2024-04-14')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Display the first few rows of the data
print(stock_data)

# Save DataFrame to a CSV file
stock_data.to_csv('C:/Users/lenovo/Desktop/Spirit Airlines.csv')


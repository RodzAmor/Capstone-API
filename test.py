import os

def extract_tickers(directory):
    tickers = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                ticker = file.replace('.csv', '')
                tickers.append(ticker)
    return tickers

# Replace 'path_to_directory' with your actual directory path
# directory = './companies'
# directory = './companies/Energy'
# all_tickers = extract_tickers(directory)
# print(all_tickers)

def find_csv_directory(ticker):
    for root, dirs, files in os.walk('./companies'):
        for file in files:
            if file == f"{ticker.upper()}.csv":
                return root
    return None 

print(find_csv_directory("AAPL"))
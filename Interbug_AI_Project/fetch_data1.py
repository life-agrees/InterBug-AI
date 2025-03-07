import requests
import pandas as pd 

import requests
import pandas as pd

# Set up request headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
}

# Define the API endpoint
base_url = 'https://api.seistream.app'

urls = {
    'blocks': f'{base_url}/blocks',
    'cosmos_transactions': f'{base_url}/transactions',
    'cosmos_statistics': f'{base_url}/contracts/cosmos/statistics',
    'cosmos_contracts': f'{base_url}/contracts/cosmos',
    'cosmos_new_block': f'{base_url}/metrics/cosmos/blocks/new?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'cosmos_avg_block_size': f'{base_url}/metrics/cosmos/blocks/avgsize?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'cosmos_avg_trans_fee': f'{base_url}/metrics/cosmos/transactions/avgfee?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'cosmos_total_trans': f'{base_url}/metrics/cosmos/transactions/fee?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'cosmos_new_trans': f'{base_url}/metrics/cosmos/transactions/new?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'cosmos_avg_gas_used': f'{base_url}/metrics/cosmos/transactions/avggasused?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'cosmos_total_gas_used': f'{base_url}/metrics/cosmos/transactions/gasused?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_transactions': f'{base_url}/transactions/evm',
    'evm_statistics': f'{base_url}/contracts/evm/statistics',
    'evm_contracts': f'{base_url}/contracts/evm',
    'evm_new_block': f'{base_url}/metrics/evm/blocks/new?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_avg_block_size': f'{base_url}/metrics/evm/blocks/avgsize?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_avg_trans_fee': f'{base_url}/metrics/evm/transactions/avgfee?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_total_trans': f'{base_url}/metrics/evm/transactions/fee?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_new_trans': f'{base_url}/metrics/evm/transactions/new?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_trans_success_rate': f'{base_url}/metrics/evm/transactions/success?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_trans_growth': f'{base_url}/metrics/evm/transactions/growth?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_avg_gas_limit': f'{base_url}/metrics/evm/gas/avglimit?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024',
    'evm_gas_growth': f'{base_url}/metrics/evm/gas/growth?date_from=10%2F1%2F2024&date_to=11%2F1%2F2024'
}

# Common parameters for all requests
params = {
    'limit': 50,
    'offset': 0
}

# Initialize a dictionary to store data for each key
all_data = {}

for key, url in urls.items():
    try:
        # Fetch the response
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            
            # Parse data
            if isinstance(data, dict):
                if 'items' in data:
                    data_to_use = data['items']  # Use 'items' list if available
                elif 'results' in data:
                    data_to_use = data['results']
                elif 'data' in data:  # Check for 'data' key in some APIs
                    data_to_use = data['data']
                else:
                    data_to_use = data  # Fallback to the raw dictionary
            else:
                data_to_use = data  # If the response is a flat list

            # Ensure data_to_use is a list for DataFrame compatibility
            if isinstance(data_to_use, dict):
                data_to_use = [data_to_use]

            # Convert to DataFrame
            df = pd.DataFrame(data_to_use)
            if not df.empty:
                # Extract 'value' column if it exists
                if 'value' in df.columns:
                    all_data[key] = df['value']
                else:
                    all_data[key] = df.iloc[:, 0]  # Use the first column as data
            
            print(f"Fetched data for {key}: {len(data_to_use)} records.")
        else:
            print(f"Failed to fetch {key} data:", response.status_code, response.text)

    except Exception as e:
        print(f"Error fetching {key} data:", e)

# Combine all data into a single DataFrame
combined_df = pd.concat(all_data, axis=1)

# Save to a single CSV file
combined_df.to_csv('combined_datas.csv', index=False)
print("Combined data saved to 'combined_data.csv'.")





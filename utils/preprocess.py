import pandas as pd

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Rename columns for consistency
    df = df.rename(columns={
        'price': 'Close',
        'total_volume': 'Volume'
    })

    # Keep only required columns
    df = df[['Close', 'Volume']]

    # Feature engineering
    df['return'] = df['Close'].pct_change()
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['volume_change'] = df['Volume'].pct_change()

    # Target: 1 if next price goes up, else 0
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop missing values
    df = df.dropna()

    return df


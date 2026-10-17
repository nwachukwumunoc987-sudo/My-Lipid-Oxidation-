import pandas as pd


def load_and_organize_data(filepath):
  """Loads refinery batch data and handles ordering."""
  df = pd.read_csv(filepath)

  # Option A: Keep the exact order the rows were entered in your file
  print("Data loaded in original sequence.")

  # Option B: If you want to explicitly sort chronologically by date and time
  # df['date'] = pd.to_datetime(df['date'])
  # df = df.sort_values(by=['date', 'time_of_sample_collection'], ascending=True)

  return df


if __name__ == "__main__":
  # Load the data file matching your In-Process Assessment Sheet structure
  data = load_and_organize_data("../data/refinery_batch_data.csv")
  print(data.head())

import pickle
import pandas as pd

columns = ["LX", "LY", "RX", "RY"]
data = [0,0,0,0]
df = pd.DataFrame( columns = columns)
df.loc[0] = data
print(df)

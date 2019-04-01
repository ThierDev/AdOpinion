#import pandas as pd
import sys
import pandas as pd

data = pd.read_csv('/home/thierry/Documents/data_science/routes/output.csv')
with open('/home/thierry/Documents/data_science/routes/positiveOutput.csv', 'w') as f1:
    with open('/home/thierry/Documents/data_science/routes/negativeOutput.csv', 'w') as f2:
        for i in range(0,len(data)-1):
            if (data["label"][i] == 0):
                f1.write(str(data["name"][i])+" ____ "+str(data["text"][i])+"\n")
            else:
                f2.write(str(data["name"][i])+" ____ "+str(data["text"][i])+"\n")

f1.close()
f2.close()


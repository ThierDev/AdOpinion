#import pandas as pd
import sys
import pandas as pd
import os

cwd = os.getcwd()

data = pd.read_csv(cwd+'/routes/output.csv')
with open(cwd+'/routes/positiveOutput.csv', 'w') as f1:
    with open(cwd+'/routes/negativeOutput.csv', 'w') as f2:
        for i in range(0,len(data)-1):
            if (data["label"][i] == 0):
                f1.write(str(data["name"][i])+" ____ "+str(data["text"][i])+"\n")
            else:
                f2.write(str(data["name"][i])+" ____ "+str(data["text"][i])+"\n")

f1.close()
f2.close()


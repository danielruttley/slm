import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')

df = pd.read_csv(r"Z:\Tweezer\Code\Python 3.7\slm\images\2023\February\02\Measure 61\trap_df.csv",index_col=0)
plt.scatter(df['img_x'],df['img_y'])
plt.show()

iterations = 200
data = []
for index, row in df.iterrows():
    ind_data = []
    for iteration in range(iterations):
        try:
            # print(f'I0_{iteration}')
            ind_data.append(row[f'I0_{iteration}'])
        except KeyError:
            pass
    data.append(ind_data)

for d in data:
    plt.plot(range(0,len(d)),d,'-o')
plt.xlabel('correction iteration')
plt.ylabel('fitted intensity (arb.)')
# plt.ylim(300,1000)
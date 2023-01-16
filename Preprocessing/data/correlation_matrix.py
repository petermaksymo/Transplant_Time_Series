import pandas as pd
from seaborn import heatmap
from matplotlib.pyplot import show


def absHighPass(df, labelCols, absThresh):
    passed = set()
    combo = [(r, c) for r in df.columns.tolist() for c in labelCols]
    print(combo)
    for (r,c) in combo:
        if (abs(df.loc[r,c]) >= absThresh):
            passed.add(r)
            passed.add(c)
    passed = sorted(passed)
    return df.loc[passed,passed]


df = pd.read_csv('./combined_data.csv')
dropCols = [x for x in df.columns if 'dummy' in x]
df = df.drop(columns=dropCols)

labelCols = [x for x in df.columns if '_1_yr' in x or '_5_yr' in x]
compareCols = set(df.columns[0:-13].tolist() + labelCols)
corrDF = df[compareCols].corr()

heatmap(absHighPass(corrDF, labelCols, 0.05),cmap="YlGnBu")
show()

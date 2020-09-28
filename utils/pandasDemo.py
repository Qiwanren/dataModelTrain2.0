import pandas as pd

def method01():
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002, 2003],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
    df = pd.DataFrame(data)
    print(df.shape[0])
    print(df.iloc[2:4,])

def method02():
    n = 24252125
    y = n/5000000
    print(y)
    print(int(y))
    for i in range(int(y)+1):
        print(i)


if __name__ == '__main__':
    method01()
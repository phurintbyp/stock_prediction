import matplotlib.pyplot as plt
import pandas as pd

data = {
    "2024-12-31": 17.88,
    "2023-12-31": 16.22,
    "2022-12-31": 12.44,
    "2021-12-31": 15.44,
    "2020-12-31": 8.87,
    "2019-12-31": 10.41,
    "2018-12-31": 8.87,
    "2017-12-31": 6.88,
    "2016-12-31": 6.06,
    "2015-12-31": 5.99,
    "2014-12-31": 5.29,
    "2013-12-31": 6.01,
    "2012-12-31": 5.27,
    "2011-12-31": 4.47,
    "2010-12-31": 3.6,
    "2009-12-31": 2.24,
    "2008-12-31": 1.4,
    "2007-12-31": 4.37,
    "2006-12-31": 3.86,
    "2005-12-31": 2.95,
    "2004-12-31": 3.01,
    "2003-12-31": 3.25,
    "2002-12-31": 1.24,
    "2001-12-31": 1.97,
    "2000-12-31": 3.06,
    "1999-12-31": 4.17,
    "1998-12-31": 3.03,
    "1997-12-31": 2.75,
    "1996-12-31": 2.5,
}

df = pd.DataFrame(list(data.items()), columns=["Date", "Value"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")



plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Value"], marker="o", linestyle="-", linewidth=2)
plt.xlabel("Year")
plt.ylabel("Value")
plt.title("EPS data")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

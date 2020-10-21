import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm

a = 7
print(a)


# todo finish this code
def b():
    b = 3
    print(b)


def switch_test(arg):
    day_dict = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    }
    print(day_dict.get(arg, "Invalid day"))


switch_test(0)
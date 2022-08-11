# # -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.impute import SimleImputer
# imputer = SimleImputer (missing_values = np.nan , strategy = "mean")
# plt.style.use("classic")
# plt.style.use("bmh")
# plt.style.use("seaborn")
# plt.style.use("seaborn-whitegrid")
# plt.style.use("seaborn-paper")
plt.style.use("dark_background")
# beginning of numpy
lst = [
    ["saleh", 12100514, 3.7],
    ["mario", 12100315, 3.9],
    ["Hossam", 12100568, 3],
    ["amr", 12100468, 2.7]
]
lstt = [
    ["saleh", 5],
    ["Hossam", 7],
    ["amr", 9],
    ["mario", 4]
]
lstInt = [4, 7, 5, 9, 8, 2]
lst3 = ["saleh", 12100514, 3.7]
lst1 = ["Hossam", 12100568, 3]
lst2 = ["amr", 12100468, 2.7]
lst0 = ["mario", 12100315, 3.9]
lst1 = ["Hossam", 12100568, 3]
lstStr = ["saleh", "ahmed", "mohamed", "saleh"]
array = np.array(lst)
print(lstInt)
print("-------------------------")
print(array)
print("-------------------------")
arrange = np.arange(1, 15, 2)  # print numbers from 1 to 10 two by two
print(arrange)
print("-------------------------")
zeros = np.zeros((3, 4))  # print matrix of zeros only
print(zeros)
print("-------------------------")
ones = np.ones((3, 4))  # print matrix of ones only
print(ones)
print("-------------------------")
r = np.random.rand(3, 4)  # print matrix of random numbers
print(r)
print("-------------------------")
u = np.random.randint(100, 150, 9)  # print 9 randome numbers from 100 to 150
print(u)
print("-------------------------")
# print matrix of random numbers from 100 to 150
s = np.random.randint(100, 150, (3, 6))
print(s)
print("-------------------------")
no_shape = np.random.randint(15, 70, 12)
shape = no_shape.reshape(4, 3)
print(shape)
print("-------------------------")
m = np.arange(3, 19, 3)
np.random.shuffle(m)  # shuffle "function" makes numbers randomly distributed
num_reshape = m.reshape(3, 2)
print(m, "\n")
print(num_reshape)
print("-------------------------")
num = np.random.randint(3, 60, (7, 5))
maxi = num.max()
print(num)
print("maximum = ", maxi)
maxInd = num.argmax()  # print the index of the maximum number
print("max index = ", maxInd)
mini = num.min()
minInd = num.argmin()
print("minimum = ", mini)
print("min index = ", minInd)
print("dimensions of the matrix: ", num.shape,
      ",", num.shape[0], ",", num.shape[1])
# end of numpy
print("-------------------------")

# beginning of pandas

ser = pd.Series(lstInt)  # Series function is like matrix but with one columns
ser2 = pd.Series(lstStr)
print(ser.values, "\n")
print(ser.index, "\n")
print(ser2.describe(), "\n")
# can choose one of the described by replacing it's name with describe function except percentages
print(ser.describe(), "\n")
# or use "agg" function notice that it is not used for strings
print(ser.agg(["mean", "std", "sum"]))
print("-------------------------")
# DataFrame function is like matrix, names of columns must have same numbers of items numbers of the list
frame = pd.DataFrame(
    lstInt, index=['a', 'b', 'c', 'd', 'e', 'f'], columns=['first'])
print(frame, "\n", frame.values, "\n", frame.index, "\n", frame.columns,
      "\n", frame.describe(), "\n", frame.agg(['mean', 'std']))
print("-------------------------")
frame2 = pd.DataFrame([lst3, lst1, lst0, lst2], index=[
                      "student 1", "student 2", "student 3", "student 4"], columns=['name', "id", 'gpa'])
print(frame2, "\n", frame2.loc["student 1":"student 3", :"id"], "\n"  # loc "function" get the location or columns you select by it's name and index (in string form) using ":" to determine the range of columns and indexes you want
      # iloc "function" is same as loc but you get data only by the defult indexes which is integrs
      , frame2.iloc[: 3, 1: 3], "\n", frame2.describe())
print("-------------------------")
frame3 = pd.DataFrame(lstStr, index=['a', 'b', 'c', 'd'], columns=['first'])
print(frame3.describe())
print("-------------------------")
df = pd.DataFrame(lst, columns=['name', "id", 'gpa'])
df2 = pd.DataFrame(lstt, columns=["name", "class"])
# concat "function"is used to cocatenat two lists , axis is used to choose if you want to connect them as columns or row where "1" means columns and "0" means rows
con = pd.concat([df, df2], axis=1)
# merge "function" is used to merge two columns by a common column between them must have a relation between the two columns
merge = pd.merge(df, df2)
print(df, "\n",
      df2, "\n", "---", "\n",
      con, "\n", "---", "\n",
      merge)
print("-------------------------")
# data = pd.read_("D:\projects\is project.accdb")
# print(data)
# end of pandas
print("-------------------------")

# beginning of matplotlib

area = np.array([100, 200, 300, 400, 500])
price = np.array([1000, 2000, 3000, 4000, 5000])
price2 = np.array([1600, 2400, 3700, 4300, 5500])
numOfRooms = np.array([2, 2, 4, 4, 5])
# data = np.array(area,price)
z = [1500, 2500, 3500, 4500, 5500]
e = [500, 800, 900, 700, 200]

# plt.plot(area,price, color = 'b', linewidth=3, linestyle="solid") # or "-"
# plt.plot(z,e, color = 'r', linewidth=4, linestyle="dashed") # or "--"
# plt.plot(e,z, linewidth=3, linestyle="dashdot") # or "-."

# plt.plot(e,z, linewidth=3, linestyle="dotted") # or ":"
# some examples of "linestyle"
# plt.plot(e,z, color = 'b', linewidth=3, linestyle="dashed") # or "--"
# plt.plot(z,e, color = 'b', linewidth=3, linestyle="dashdot") # or "-."
# plt.plot(e,z, color = 'b', linewidth=3, linestyle="dotted") # or "."

# plt.title("Area VS Price") #title of the graph
# plt.xlabel("Area")
# plt.ylabel("Price")
design = plt.axes()  # or you can use
design.set(title="Area VS Price", xlabel="Area", ylabel="Price")
plt.xlim(50, 700)  # limits of x-axis
plt.ylim(500, 8000)  # limits of y-axis

plt.plot(area, price, "-b", label="Tower 1")
plt.plot(area, price2, "--r", label="Tower 2")

x = np.random.randint(100, 1000, 9)
y = np.random.randint(1000, 10000, 9)

plt.legend()  # allow you to know the linestyle of each plot
# scatter "function" put a point at each intersection point between x-axis and y-axes "c refers to color , m for shape of the point and s for size"
plt.scatter(x, y, marker="o", c="w", s=50)
plt.show()  # print the plot  "to print two plot separetly use "show" function after each plot"

# subplot "functiom" allow you to print plots in the same screen writtten like matrix
plt.subplot(2, 1, 1)
plt.plot(area, price)
plt.subplot(2, 1, 2)
plt.plot(area, price2)

plt.show()

plt.hist(array,  # data in the first line
         histtype="bar",  # there is more types "barstacked","step","stepfilled"
         alpha=1,  # transparency
         bins=5  # numbers of bars
         )
plt.show()

carType = np.array(["audi", "mercedec", "kia", "BMW"])
carNum = np.array([3, 5, 1, 4])
# bar "function" get x , y axis inform of array or just numbers and print them inform of bars
plt.bar(carType, carNum, color="y", alpha=0.9)
plt.show()

plt.pie(  # pie "function" print data in form of pie "فطيرة" shape
    carNum, labels=carType,
    # if you want to specify a specific data be cutting it
    explode=[0, 0.1, 0, 0],
    radius=1.5,  # the radius of the pie
    startangle=180,  # angele you want to start by
    colors=['w', 'y', 'b', 'r'],  # must be the same number of data
    autopct="%1.0f%%",
)
plt.show()

# axes "function" give the projection in order to be used in plot3D "function"
axis = plt.axes(projection="3d")
# print 3d plot after getting it's projection from axes "function"
axis.plot3D(area, price, numOfRooms, 'b')
# same as normal scatter but print in form of 3d
axis.scatter3D(area, price, numOfRooms, c='r')
plt.show()

# end of matplotlip
print("-------------------------")

# beginning of seaborn

Data = pd.read_csv("StudentsPerformance.csv")

correlation = Data.corr()  # corr "function" print the correlation between tables of data in form of numbers "-1 to 1" where +ve means directly relationship and -ve means inversly relationship
# print relations between tables of data in form of heatmab
sns.heatmap(correlation)
# print relations between tables of data in form of plots and scatters
sns.pairplot(Data)
plt.show()


print(Data, "\n", "---", "\n",
      Data.shape, "\n", "---", "\n",
      Data.dtypes, "\n", "---", "\n",
      # Data.unique(),"\n",
      Data.nunique()
      )

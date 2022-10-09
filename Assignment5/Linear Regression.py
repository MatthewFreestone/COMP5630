import pandas as pd                                
import numpy as np       
import matplotlib.pyplot as plt


data = pd.read_csv("numpydataset.csv")      # load the csv file as a DataFrame
data.head()                                 # displays the first 5 rows in the dataset


samples = len(data)              # calculating number of samples


def MSE(points, m, b):
    # write your code here...
    x, y = points
    print(x,y)
    pred = m * x + b
    return np.mean((y-pred)**2)
    
    #return mean_sqaured_error


def gradient_descent(m_current, b_current, points, step_size):
    return m_current, b_current
    # write your code here...
    
    #return m_new, b_new




m, b = 0, 0
L = 0.001       # initial learning rate, can be adjusted later
epochs = 1    # we iterate over the same dataset 100 times
# epochs = 100    # we iterate over the same dataset 100 times

for epoch in range(1, epochs+1):
    m, b = gradient_descent(m, b, data, L)
    loss = MSE(data, m, b)
    print(f"Epoch {epoch}, m: {m}, b:{b}, Loss: {loss}")
print(m, b, loss)



fig, ax = plt.subplots(1,1)

ax.scatter(data.Features, 
           data.Targets, 
           color="red", 
           linewidths=0.5, 
           label="Points")
ax.plot(data.Features, 
        [m * x + b for x in data.Features], 
        linewidth=3, 
        linestyle="dashed", 
        label="$ f(x) = mx+c $")

ax.legend(loc="lower right", bbox_to_anchor=(.96, 0.0))
ax.set_xlabel("Features")
ax.set_ylabel("Targets")

plt.savefig('LinearRegression001.png')

plt.close()


m, b = 0, 0
L = 0.01   # new learning rate
epochs = 100

for epoch in range(1, epochs+1):
    m, b = gradient_descent(m, b, data, L)
    loss = MSE(data, m, b)
    print(f"Epoch {epoch}, m: {m}, b:{b}, Loss: {loss}")
print(m, b, loss)



fig, ax = plt.subplots(1,1)

ax.scatter(data.Features, 
           data.Targets, 
           color="red", 
           linewidths=0.5, 
           label="Points")
ax.plot(data.Features, 
        [m * x + b for x in data.Features], 
        linewidth=3, 
        linestyle="dashed", 
        label="$ f(x) = mx+c $")

ax.legend(loc="lower right", bbox_to_anchor=(.96, 0.0))
ax.set_xlabel("Features")
ax.set_ylabel("Targets")

plt.savefig('LinearRegression01.png')
plt.close()


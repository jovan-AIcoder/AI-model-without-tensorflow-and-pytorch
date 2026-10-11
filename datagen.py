import math
import random
X = []
y = []
max_value = float(input('Max Value: '))
res = float(input('Resolution: '))
i = 0
while(i <= max_value):
    X.append(i)
    y.append(1.0 - math.exp(-i) + random.normalvariate(0, 0.01))
    i += res

dataset = ""
for i in range(len(X)):
    dataset += f"{X[i]},{y[i]}\n"

with open('dataset.txt', 'w') as file:
    file.write(dataset)

print('Done!')
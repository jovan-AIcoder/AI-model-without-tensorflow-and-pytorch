import matplotlib.pyplot as plt
import neural
import joblib
# load dataset
with open('dataset.txt', 'r') as file:
    data = file.read()
dataset = data.split('\n')
df = []
for i in range(len(dataset)):
    df.append(dataset[i].split(','))
X = []
y = []
for i in range(len(df) - 1):
    X.append(float(df[i][0]))
    y.append(float(df[i][1]))

model = joblib.load('model.pkl')
y_pred = []
for i in range(len(X)):
    y_pred.append(model.predict(X[i]))

plt.figure(figsize=(10,9))
plt.scatter(X, y, c='blue')
plt.plot(X, y_pred, color="#FF8400", linestyle='--')
plt.title('My First DIY AI Model')
plt.xlabel('X')
plt.ylabel('y')
plt.savefig('plot.png')
plt.show()
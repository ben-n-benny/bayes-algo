import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt 
import seaborn as sns 


# Load the mushroom dataset into a Pandas DataFrame
df = pd.read_csv("mushrooms.csv")
df = df.astype('category')


encoder = LabelEncoder()

for col in df.columns:
  df[col] = encoder.fit_transform(df[col])

df = df.drop('VEIL-TYPE', axis=1)

x = df.drop(['CLASS'], axis=1)
#x = x.to_numpy()
y = df['CLASS']
y = y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=30)

nb = GaussianNB()
nb.fit(x_train, y_train)

print("\nTest Accuracy: {}%".format(round(nb.score(x_test, y_test)*100, 2)))

print("\nEdible count: ", np.count_nonzero(y == 0))
print("Poisonous count: ", np.count_nonzero(y == 1))

print("\nNaive Bayes Classifier report: \n\n", classification_report(y_test, nb.predict(x_test)))

print(nb.predict(x))
print(y)

cap_shape = int(input("Enter the cap shape"))
surface = int(input("Enter the surface "))
color = int(input("Enter the color"))
bruises = int(input("Does the mushroom have bruises "))
odor = int(input("Enter the odor"))
gill_attach = int(input("Enter the gill attachment"))
gill_spacing = int(input("Enter the gill spacing"))
gill_size = int(input("Enter the gill size"))
gill_color = int(input("Enter the gill color "))
stalk_shape =int( input("Enter the stalk shape"))
stalk_root = int(input("Enter the stalk root"))
stalk_surf_above_ring = int(input("Enter the stalk surface above ring"))
stalk_surf_below_ring = int(input("Enter the stalk surface below ring"))
stalk_color_above_ring = int(input("Enter the stalk color above ring "))
stalk_color_below_ring = int(input("Enter the stalk color below ring "))
veil_color = int(input("Enter the veil color"))
ring_number = int(input("Enter the number of rings"))
ring_type = int(input("Enter the ring type"))
spore_print_color = int(input("Enter the spore print color"))
population = int(input("Enter the population "))
habitat = int(input("Enter the habitat "))

input_list = [cap_shape, surface, color, bruises, odor, gill_attach, gill_spacing, gill_size, gill_color, stalk_shape, stalk_root, stalk_surf_above_ring, stalk_surf_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type, spore_print_color, population, habitat]

input_array = np.array(input_list).reshape(1, -1)
new_mushroom = np.array(input_array)

predicted_class = nb.predict(new_mushroom)
print(f"\nPredicted class: {predicted_class[0]}")
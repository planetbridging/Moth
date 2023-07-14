import pandas as pd

# Array of numbers to check
numbers_to_check = [171]
start = 172
end = 337

number_list2 = numbers_to_check + list(range(start, end+1))
number_list = number_list2 + list(range(446, 499))
print(number_list)

# Read the CSV file
data = pd.read_csv('data2.csv')

# Update the values in the first column based on line numbers
data['MeFound'] = data.index.map(lambda x: False if x+1 in number_list else data['MeFound'].iloc[x])

# Save the modified data back to the CSV file
data.to_csv('data2.csv', index=False)

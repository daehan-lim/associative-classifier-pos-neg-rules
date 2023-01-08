array = [1, 2, 3, 4, 5]
array[0] = 9
print(array)

array2 = array
array2[0] = 2
print(array2)
print(array)

# Create a sample collection
users = {'Hans': 'active', 'Éléonore': 'inactive', '景太郎': 'active'}

print(users.copy().items())
# Strategy:  Iterate over a copy
for user, status in users.copy().items():
    if status == 'inactive':
        del users[user]

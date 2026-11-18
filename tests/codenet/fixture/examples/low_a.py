L = input()
L = L.split()
for i, val in enumerate(L):
    L[i] = int(val)
t = L[0]/L[1]
t = format(t, '0.4f')
print(t)

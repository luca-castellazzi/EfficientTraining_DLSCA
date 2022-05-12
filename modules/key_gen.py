# Random generation of the 3 keys used in MDM

import random

random.seed(24)

l1 = []
l2 = []
l3 = []

for _ in range(16):
    r1 = hex(random.randint(0, 255)).replace('0x', '')
    r2 = hex(random.randint(0, 255)).replace('0x', '')
    r3 = hex(random.randint(0, 255)).replace('0x', '')
    
    if len(r1) == 1:
        r1 = f'0{r1}'
    if len(r2) == 1:
        r2 = f'0{r2}'
    if len(r3) == 1:
        r3 = f'0{r3}'

    l1.append(r1)
    l2.append(r2)
    l3.append(r3)

k1 = ''.join(l1)
k2 = ''.join(l2)
k3 = ''.join(l3)

print(f'k1 = {k1}')
print(f'k2 = {k2}')
print(f'k3 = {k3}')

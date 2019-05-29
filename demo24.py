import copy
from copy import deepcopy

leftHand = ['A', 'K']
rightHand = ['Q', 'J']
user1 = [leftHand, rightHand]

# 指向同一個記憶體空間
user2 = user1

# 複製放在新的記憶體空間
user3 = copy.copy(user1)

# 深度複製，就像是巢狀複製。使用全新記憶體空間來複製
user4 = copy.deepcopy(user1)


def displayList(message):
    print(message)
    print('user1:', user1)
    print('user2:', user2)
    print('user3:', user3)
    print('user4:', user4)
    print('user1:', hex(id(user1)), 'user2:', hex(id(user2)))
    print('user3:', hex(id(user3)), 'user4:', hex(id(user4)))


displayList("first time")
leftHand[0] = '10'
displayList("second time")
user1.append('joker')
displayList("third time")

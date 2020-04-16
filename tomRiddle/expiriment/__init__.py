


def funct():
    print("funct runing")


list = []
dict = {}

print(type(funct))
print(type(dict))
print(type(list))

if type(funct).__name__ == 'function':
    print("this is a function")
    funct()
else:
    print("don't get it")
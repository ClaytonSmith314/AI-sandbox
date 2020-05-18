import threading

def threadfunction(z):
    x = 0
    while x <= z:
        x = x+1
        print(x)


thread1 = threading.Thread(target=threadfunction, args=(20,))

thread1.start()

print("waiting for thread1 to finish")
print("how long now...")
a = 0
while a <= 10:
    a = a + 1
    print('...')
print("is it done yet?")

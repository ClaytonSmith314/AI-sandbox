
import Memory

__init_code = 0


# code 0, default start
def boot0():
    print("starting default boot")


def close0():
    Memory.dump()


# methods implemented outside of setup.py
def boot(code = 0):
    global __init_code

    boot_codes = {
        0: boot0
    }

    __init_code = code
    print("starting boot procedure from code " + str(code))
    boot_codes[code]()


def close(code=__init_code):
    print("\n\nclosing from code " + str(code))
    close_codes = {
        0: close0
    }

    close_codes[code]()

    print("successfully closed")

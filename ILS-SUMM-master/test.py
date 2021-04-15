def oui():
    print("oui")

def non():
    print("non")

def switch(y):
    switcher = {
        '1': oui,
        '2': non,
    }

    func = switcher.get(y, lambda: 'gros noob')
    return func()

if __name__ == '__main__':
    switch('1')
    switch('1')
    switch('2')


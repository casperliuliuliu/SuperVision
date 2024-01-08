from datetime import datetime
from RunModels.train_config import get_config


def pprint(output = '\n' , filename = get_config()['log_file_path'], show_time = False):
    
    print(output)
    with open(filename, 'a') as f:
        if show_time:
            print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), file=f)
        print(output, file=f)
        
if __name__ == "__main__":
    # usual string
    pprint('hello')
    var = 10
    dic = {'a': 'apple', 'b': 'ball'}
    # using f-string
    pprint(f'var = {var}')
    pprint(f'dic = {dic}')
    # using str.format
    pprint('var is = {:3d}'.format(var))
    pprint('dic a is {a}, b is {b}'.format(**dic))
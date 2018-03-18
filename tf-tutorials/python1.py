
def arg1(*arg):
    sum=0
    for i in arg:
        sum+=i 
        
    return sum 


print arg1(1,2,3,4,5)


def print_everything(*args):
    for count, thing in enumerate(args):
            print '{0}. {1}'.format(count, thing)

print_everything('apple', 'banana', 'cabbage')


def unwrap_dict(name,country):
    
    print name," ",country
    
names=['milton','bd']
unwrap_dict(*names)


def print_three_things(a, b, c):
  print 'a = {0}, b = {1}, c = {2}'.format(a,b,c)

mylist = ['aardvark', 'baboon', 'cat']
print_three_things(*mylist)

def table_things(**kwargs):
    for name, value in kwargs.items():
        print '{0} = {1}'.format(name, value)

table_things(apple = 'fruit', cabbage = 'vegetable')


class Foo(object):
    def __init__(self, value1, value2):
        # do something with the values
        print value1, value2

class MyFoo(Foo):
    def __init__(self, *args, **kwargs):
        # do something else, don't care about the args
        print 'myfoo'
        super(MyFoo, self).__init__(*args, **kwargs)
        

def arg2(*arg1,**arg2):
    print "arg1 count ",len(arg1)
    
    print "arg2 count ", len(arg2)
    
    
    
arg2(*[1,2,3],**{'a':'h','b':'b'})

print "{} {}".format('bd',1)
    
    
    
    
        
        

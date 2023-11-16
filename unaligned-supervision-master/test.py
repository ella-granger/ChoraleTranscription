from sacred import Experiment

ex = Experiment("config_test")

@ex.config
def cfg():
    a = 10
    b = 3 * a
    c = "foo"

@ex.automain
def my_main(a,b,c,d,e):
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    

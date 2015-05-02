function testaxor(w1,w2)

a = 1.7159;
b = 2.0/3.0;

x=[0;0]
y=forward(w1,w2,x,a,b)
x=[0;1]
y=forward(w1,w2,x,a,b)
x=[1;0]
y=forward(w1,w2,x,a,b)
x=[1;1]
y=forward(w1,w2,x,a,b)

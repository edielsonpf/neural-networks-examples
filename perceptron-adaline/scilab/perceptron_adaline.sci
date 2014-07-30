function [w,epoch]=perceptron_adaline(x,w,d)
//Inputs:
//		x: vector with training data. The inputs must be distributted in rows and the input sets in lines
//		w: vector with initial synaptic weight values
//		d: vector with the expected output values
//Output:
//      w: vector with resulted synaptic weight values
//      epoch: numer of training epoch

//n: learning factor
n=0.01;
limiar=0.0001;

[Input_Size,Samples_Size]=size(x)

HistError=[];
epoch=0;
Error=eqm(x,w,d);

while Error > limiar then
    
    LastError=eqm(x,w,d);
    for k=1:Samples_Size
        v=w'*x(:,k);
       //Calculating the error in the i instant
        e=d(k)-v;
        w=w+n*e.*x(:,k);
    end
    epoch=epoch+1;    
    HistError(epoch)=eqm(x,w,d);
    Error=abs(HistError(epoch)-LastError);
end
plot(HistError);
endfunction

function [w,epoch]=perceptron_hebbian(x,w,d)
//Inputs:
//		x: vector with training data. The inputs must be distributted in rows and the input sets in lines
//		w: vector with initial synaptic weight values
//		d: vector with the expected output values
//Output:
//      wR: vector with resulted synaptic weight values
//      epoch: numer of training epoch

//n: learning factor
n=0.01;
ERROR_EXIST = 1;
ERROR_INEXIST = 0;

[Input_Size,Samples_Size]=size(x)


errors=ERROR_EXIST;
epoch=0;

while errors == ERROR_EXIST then
    errors = ERROR_INEXIST;
    //disp("Starting training epoch: "+string(epoch));
    //Treining the system for all L sample training data
    for k=1:Samples_Size
        v=w'*x(:,k);
        //Activation function-------------------------
        y=sign(v);
        // if error exist
        if y~=d(k) then
            //Calculating the error in the i instant
            e=d(k)-y;
            dw=n*e.*x(:,k);
            w= w + dw;
            errors=ERROR_EXIST;
        end
    end
    epoch=epoch+1;
end
endfunction

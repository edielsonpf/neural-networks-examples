function [w, epoch]=perceptron_one_epoch(x,w,d)
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

//[Number_samples,Input_Size]=size(x)
[Input_Size, Number_samples]=size(x);
disp("Input size: "+string(Input_Size));
disp("Number of samples: "+string(Number_samples));

errors=ERROR_EXIST;
epoch=0;
    
while errors == ERROR_EXIST then
    errors = ERROR_INEXIST;
    //disp("Starting training epoch: "+string(epoch));
    //Treining the system for all L sample training data
    [a,b]=create_linear_separation(x,w);
    plot(a,b);
    for k=1:Number_samples
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
    disp("Updated synaptic weightd:")
    disp(w);
    halt();
end
   [a,b]=create_linear_separation(x,w);
    plot(a,b);
endfunction

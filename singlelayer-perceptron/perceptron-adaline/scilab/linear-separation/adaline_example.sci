function adaline_example()

    //A_x = grand(5, 1, "nor", 2, 1);
    //A_y = grand(5, 1, "nor", 3, 1);
    //A=[A_x, A_y];
    A=[2.0345522 3.4790581; 2.0081632 2.5116644];
    //disp(A);    

    //B_x = grand(5, 1, "nor", 8, 1);
    //B_y = grand(5, 1, "nor", 5, 1);
    //B=[B_x, B_y];
    B=[6.8653232, 5.3456417; 8.1727731, 4.2539951];
    //disp(B);
    
    //plot(A(:,1), A(:,2),'o');    
    //plot(B(:,1), B(:,2),'x');
    
    //training_set=[-1, A(1,:); -1, B(1,:); -1, A(2,:); -1, B(2,:); -1, A(3,:); -1, B(3,:); -1, A(4,:); -1, B(4,:); -1, A(5,:); -1, B(5,:)]';
    training_set=[-1, A(1,:); -1, B(1,:); -1, A(2,:); -1, B(2,:)]';
    disp("Training set:");
    disp(training_set);
    
    //expcted_result=[-1.0000;1.0000;-1.0000;1.0000;-1.0000;1.0000;-1.0000;1.0000;-1.0000;1.0000];
    expcted_result=[-1.0000;1.0000;-1.0000;1.0000];
    
    [r,c]=size(training_set);
    W=rand(r,1,'uniform');
    disp("Initial synaptic weight:");
    disp(W);
    
   //[x,y]=create_linear_separation(training_set,W);
   //plot(x,y);
    
    [W,epoch]=perceptron_adaline(training_set,W,expcted_result);
    disp("Final epoch:"+string(epoch));
    disp("Final synaptic weight:");
    disp(W);
endfunction

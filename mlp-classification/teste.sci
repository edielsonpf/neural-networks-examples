function y=teste(x,W)
    //Definindo arquitetura da rede
    NeuralNetwork=[4 15 3];
    y = ann_FF_run(x,NeuralNetwork,W);
    y=round(y);
    
    [l,c]=size(y);
    
    for i=1:c
        if(y(1,i)==0 & y(2,i)==0 & y(3,i)==1)
            classe='C';
        elseif (y(1,i)==1 & y(2,i)==0 & y(3,i)==0)
            classe='A';
        elseif (y(1,i)==0 & y(2,i)==1 & y(3,i)==0)
            classe='B';      
        else
            classe='indefinida';
        end 
    disp(classe);         
    end
endfunction

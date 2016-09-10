function erro = eqm(x,w,d)
    
    [Input_Size,p]=size(x);    
    erro=0;
    for k=1:p
         v=w'*x(:,k);
         erro=erro+(d(k)-v).^2;
    end
    erro=erro/p;
endfunction

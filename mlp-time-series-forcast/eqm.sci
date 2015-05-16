function erro = eqm(y,d)
    
    [Input_Size,p]=size(y);    
    erro=0;
    for k=1:p
        erro=erro+(d(k)-y(k)).^2;
    end
    erro=erro/p;
endfunction

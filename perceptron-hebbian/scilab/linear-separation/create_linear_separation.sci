function [x1,x2]=create_linear_separation(data,W)
    
    x=min(data):0.2:max(data);
    x1=(-W(3)/W(2)).*x + (W(1)/W(2));
    x2=(-W(2)/W(3)).*x + (W(1)/W(3));
    x1=x;
endfunction

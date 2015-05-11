%Definiçao de variaveis
in =[0 0;0 1;1 0;1 1];
d = [1 -1;-1 1;-1 1;1 -1]; %porta AND
d= 0.8*d';
t = size(d);
namostras=t(2); %numero de amostras e dado pelo numero de colunas de d

%w = rand(2,3);
w = [0.2 0.3 0.2;0.1 0.4 0.5];

eta = 0.7;
erro = 1;
limiar = 0;
cont =0;
%epoca
while(erro>limiar) %enquanto erro maior que limiar entra na estrutura
   %treinamento
   for i=1:namostras
      x = [1 in(i,:)];
      y = tanh(x*w')';
      
      erro = d(:,i)-y;
      
      w=w - eta*erro.*(1-y.*y)*x
      
   end; %fim de treinamento
  
   %verificaçao
   erro = 0;
   for i=1:namostras
       
      x = [1 in(i,:)];
      y = tanh(x*w')';
      
      S1 = ( d(1,i) > d(2,i) ) & (y(1) < y(2) )
      
      S2 = ( d(1,i) < d(2,i) ) & (y(1) > y(2) )
      
      if (S1 & S2)
         erro = erro+1;
      end;
         
     erro
     %x
     y
     d
     w
     
     %pause
   end; %fim de verificaçao
   
   cont = cont +1
end;

         
      

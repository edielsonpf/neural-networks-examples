function [wR,yR,epocas]=perceptron-hebbian(x,w,d,xt,dt)
%Inputs:
%		x: vector with data training
%		w: vector with initial synaptic weight values
%		d: 
%n: fator de aprendizagem
n=0.7;
limiar = 0.1;
[L,C]=size(x)
wR=w;
erros=1;
epocas=0;
%for k=1:N

while erros>limiar
    %--------------------------------------------
    epocas=epocas+1;
    %Treina o Sistema
    for i=1:L
        soma=0;
        for j=1:C
            soma=soma+x(i,j)*wR(j);
        end
        %Função ativação-------------------------
        v(i)=soma;
        
        if(v(i)>=0)
            y(i)=0.9;
        else
            y(i)=-0.9;
        end
        %----------------------------------------
	
	%Calculo do erro no instante i-----------
        e(i)=d(i)-y(i);
        %e(i)=y(i)-d(i);
        %----------------------------------------
        
	%Aprendizegem por correção de erros------
	for j=1:C
            dw=n*e(i)*x(i,j);
            wR(j)= wR(j) + dw;
        end
	%----------------------------------------
    end
    
    [Lt,Ct]=size(xt)
    %Testa a Saida
    for i=1:Lt
        soma=0;
        for j=1:Ct
            soma=soma+xt(i,j)*wR(j);
        end
        rR(i)=soma
        
        if(rR(i)>=0)
            yR(i)=0.9;
        else
            yR(i)=-0.9;
        end
        
        eR(i)=rt(i)-yR(i)
        %eR(i)=yR(i)-r(i);
        
        erros=0;
        for i=1:length(eR)
            erros=erros+abs(eR(i));
        end
    end
    %--------------------------------------------
    wR=wR
    yR=yR;
    erros=erros;
    epocas=epocas
    pause

end

clear all
% Dados da rede neural
nEntradas = 3; % numero de sinais de entrada
nEscondida = 100; % numero de neuronios da camada de entrada
nSaidas = 1; % numero de neuronios da camada de saida

% Gerando pares de treinamento
N = 1000; % numero de pares de treinamento
x=uniform_rnd(-pi/2,pi/2,nEntradas,N);
d=zeros(1,N); % saidas desejadas
for i=1:N
   d(i)=(sin(x(1,i))+sin(x(2,i))+sin(x(3,i)))/3.0;
end  

% Gerando pares de validacao cruzada
N1 = 1000; % numero de pares de treinamento
x1=uniform_rnd(-pi/2,pi/2,nEntradas,N);
d1=zeros(1,N); % saidas desejadas
for i=1:N
   d1(i)=(sin(x1(1,i))+sin(x1(2,i))+sin(x1(3,i)))/3.0;
end  

% Gerando pares de teste
N2 = 1000; % numero de pares de treinamento
x2=uniform_rnd(-pi/2,pi/2,nEntradas,N);
d2=zeros(1,N); % saidas desejadas
for i=1:N
   d2(i)=(sin(x2(1,i))+sin(x2(2,i))+sin(x2(3,i)))/3.0;
end  

[w1,w2] = treina(x,d,x1,d1,nEntradas,nEscondida,nSaidas);
    
% Testando a rede neural
for i=1:N2
   err(i) = (d2(i)-forward(x2(:,i),w1,w2)).^2;
end

media = mean(err)
maximo = max(err)
minimo = min(err)






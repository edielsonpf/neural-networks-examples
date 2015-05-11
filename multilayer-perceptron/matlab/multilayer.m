function multilayer()
clear all

% Algumas variaveis
nTreinamento = 1000000; % numero de pares p/ treinamento
nValidacao = 5000;    % numero de pares p/ validacaoo cruzada
nTeste = 5000;        % numero de pares p/ teste
maxEpocas = 500;      % numero maximo de epocas de treinamento
 
% Arquitetura da rede
nEntradas = 3;
nEscondida = 40;
nSaidas = 1;
a = 1.7159; 
b = 2/3;

% Passos de aprendizagem 
etaSaida = 0.01;
etaEscondida = 0.05;

% Gerando pares de treinamento
xt = unifrnd(-2,2,nTreinamento,nEntradas);
dt = zeros(nTreinamento,nSaidas);
for i=1:nTreinamento
  dt(i) = 1/3*(sin(xt(i,1))+sin(xt(i,2))+sin(xt(i,3)));
end

% Gerando pares de validacao cruzada
xv = unifrnd(-2,2,nValidacao,nEntradas);
dv = zeros(nValidacao,nSaidas);
for i=1:nValidacao
  dv(i) = 1/3*(sin(xv(i,1))+sin(xv(i,2))+sin(xv(i,3)));
end

% Gerando pares de teste
xteste = unifrnd(-2,2,nTeste,nEntradas);
dteste = zeros(nValidacao,nSaidas);
for i=1:nTeste
  dteste(i) = 1/3*(sin(xteste(i,1))+sin(xteste(i,2))+sin(xteste(i,3)));
end

% Treinando a rede
[w1 w2] = backpropagation(nEntradas,nEscondida,nSaidas,xt,dt,xv,dv,maxEpocas,etaSaida,etaEscondida,a,b);

% Testando a rede
erros = zeros(1,nTeste);
for i=1:nTeste
  y = forward(w1,w2,a,b,xteste(i,:));       % calculando saida da rede
  erros(i) = dteste(i,:)-y;
end 
figure(2);
hist(erros);
end


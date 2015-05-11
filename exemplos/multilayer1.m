clear all

% Algumas variaveis
nTreinamento = 3; % numero de pares de treinamento
nValidacao = 3;    % numero de pares p/ validacaoo cruzada
maxEpocas = 1000;      % numero maximo de epocas de treinamento
 
% Arquitetura da rede
nEntradas = 2;
nEscondida = 4;
nSaidas = 2;
a = 1.7159; 
b = 2/3;

% Passos de aprendizagem 
etaSaida = 0.7;
etaEscondida = 0.5;

% Gerando pares de treinamento
xt = [5 2;5 2;5 2]
dt = [0.4 -0.3;0.4 -0.3;0.4 -0.3]

% Gerando pares de validacao cruzada
xv = [5 2;5 2;5 2]
dv = [0.4 -0.3;0.4 -0.3;0.4 -0.3]

% Treinando a rede
[w1 w2] = backpropagation(nEntradas,nEscondida,nSaidas,xt,dt,xv,dv,maxEpocas,etaSaida,etaEscondida,a,b);


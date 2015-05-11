clear all
% Dados da rede neural
nEntradas = 2; % numero de sinais de entrada
nEscondida = 100; % numero de neuronios da camada de entrada
nSaidas = 1; % numero de neuronios da camada de saida

% Gerando pares de treinamento
xt=[0 0 1 1;0 1 0 1];
dt=[0.5 -0.5 -0.5 0.5];

% Gerando pares de validacao cruzada
xv=[0 0 1 1;0 1 0 1];
dv=[0.5 -0.5 -0.5 0.5];

% Gerando pares de teste
xe=[0 0 1 1;0 1 0 1];
de=[0.5 -0.5 -0.5 0.5];
Ne=4;

[w1,w2] = treina(xt,dt,xv,dv,nEntradas,nEscondida,nSaidas);




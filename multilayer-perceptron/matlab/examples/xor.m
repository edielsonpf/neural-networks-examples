clear all
% Dados da rede neural
nEntradas = 2; % numero de sinais de entrada
nEscondidas = 4; % numero de neuronios da camada de entrada
nSaidas = 1; % numero de neuronios da camada de saida

% Gerando pares de treinamento
xt=[0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1;0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1];
rt=[0.9 -0.9 -0.9 0.9 0.9 -0.9 -0.9 0.9 0.9 -0.9 -0.9 0.9 0.9 -0.9 -0.9 0.9];

%xt=[0 0 1 1 ;0 1 0 1 ];
%rt=[0.9 -0.9 -0.9 0.9];

% Gerando pares de validacao cruzada
xv=[0 0 1 1 0 0 1 1;0 1 0 1 0 1 0 1];
rv=[0.9 -0.9 -0.9 0.9 0.9 -0.9 -0.9 0.9];
%xv=[0 0 1 1 ;0 1 0 1];
%rv=[0.9 -0.9 -0.9 0.9];

[w1,w2] = treina(xt,rt,xv,rv,nEntradas,nSaidas,nEscondidas)

testaxor(w1,w2);

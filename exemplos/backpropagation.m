% function [w1 w2] = backpropagation(nEntradas,nEscondida,nSaidas,xt,dt,xv,dv,maxEpocas,etaSaida,etaEscondida,a,b)
% funcao para treino de uma rede neural multilayer perceptron
% com uma camada escondida.
% Os parametros de entrada sao:
% - nEntradas: dimensao do vetor de entrada
% - nEscondida: numero de neuronios na camada escondida
% - nSaidas: numero de neuronios na camada de saida
% - xt: matriz (nEntradas+1,nExemplos) com as entradas da rede para treinamento, onde nExemplos é o número de pares de treinamento.
% - dt: matriz (nSaidas,nExemplos) com as saídas desejadas para as respectivas entradas em xt.
% - xv e dv: idem a xt e dt, mas para os pares de validacao cruzada.
% - maxEpocas: número máximo permitido de épocas de treinamento
% - etaSaida: passo de aprendizagem da camada de saida
% - etaEscondida: passo de aprendizagem da camada escondida
% - a e b: parametros da funcao de ativacao: y = a*tanh(b*vj);
 
function [w1otimo w2otimo] = backpropagation(nEntradas,nEscondida,nSaidas,xt,dt,xv,dv,maxEpocas,etaSaida,etaEscondida,a,b)

% Verificando quantos exemplos de treinamento e validacao cruzada
t = size(xt);
nTreinamento = t(2);

% Criando a rede
delta = 0.001; % faixa de variacao dos pesos iniciais
w1 = uniform_rnd(-delta,delta,nEscondida,nEntradas+1);
w2 = uniform_rnd(-delta,delta,nSaidas,nEscondida+1);

%w1 = [-0.2 0.1 0.7;
%       0.3 -0.7 0.8;
%       0.5 0.3 -0.6;
%       0.4 0.5 -0.4];
%w2 = [0.1 0.3 -0.4 0.7 0.6;
%      -0.7 0.8 0.3 -0.4 0.5];
 

% Treinamento
eMedio = zeros(1,maxEpocas);
eMin = Inf;
epocas = 1;
while (epocas < maxEpocas)
  % Treinando a rede
  for i=1:nTreinamento
    [y yh] = forward(w1,w2,a,b,xt(i,:));       % calculando saida da rede
    [w1 w2] = backward(w1,w2,a,b,xt(i,:),yh,y,dt(i,:),etaSaida,etaEscondida); % atualizando pesos
  endfor   

  % Validacao cruzada
  eMedio(epocas) = valida(w1,w2,a,b,xv,dv);

  % Salvando rede de melhor desempenho
  if (eMedio(epocas) < eMin)
    eMin = eMedio(epocas);
    epocaminimo = epocas;
    w1otimo = w1;
    w2otimo = w2;
  endif

  % Mudando a ordem de apresentação dos exemplos



  epocas=epocas+1;
endwhile

% Mostrando evolucao do erro medio com as epocas
figure(1);
plot(eMedio);

["Melhor epoca: " int2str(epocaminimo)]

endfunction


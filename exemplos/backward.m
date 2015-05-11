% Funcao para atualizacao dos pesos de uma rede neural do tipo multilayer perceptron
% com uma camada escondida.
% Entradas:
% - w1,w2: pesos sinapticos da camada escondida e da camada de saida, respectivamente
% - a e b: parametros da funcao de ativacao: y = a*tanh(b*vj);
% - x: vetor de entrada da rede
% - yh: saida da camada escondida
% - y: saida da rede neural
% - d: valor desejado de saida da rede, para a entrada x
% - etaSaida: passo de aprendizagem da camada de saida
% - etaEscondida: passo de aprendizagem da camada escondida

% Saidas:
% - w1atual e w2atual: pesos sinapticos atualizados

function [w1atual w2atual] = backward(w1,w2,a,b,x,yh,y,d,etaSaida,etaEscondida)

  % Determinando erro da saida
  e = d - y;

  % Gradiente local da camada de saida
  deltaSaida = b/a*e.*(a-y).*(a+y);

  % Atualizando pesos da camada de saida
  w2atual = w2 + etaSaida*deltaSaida'*[1 yh]; 
  
  % Matriz de pesos da camada de saida da epoca anterior sem os pesos relativos ao bias
  t = size(w2);
  w2red = w2(:,2:t(2));

  % Gradiente local da camada escondida
  deltaEscondida = b/a*(a-yh).*(a+yh).*(deltaSaida*w2red);
  
  % Atualizando pesos da camada escondida
  w1atual = w1 + etaEscondida*deltaEscondida'*[1 x]; 

endfunction


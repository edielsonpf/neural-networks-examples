% Funcao que fornece a saida de uma rede neural multilayer perceptron de uma camada escondida.
% Entradas:
% - w1, w2: pesos da rede
% - a e b: parametros da funcao de ativacao: y = a*tanh(b*vj); 
% - x: vetor de entrada 
% Saidas:
% - y: saida da rede
% - yh: saida da camada escondida
% Observacao:
% se a funcao for chamada como z = forward(w1,w2,a,b,x), z recebera o valor de y.

function [y yh] = forward(w1,w2,a,b,x)

  % Adicionando entrada x0 (bias)
  xin = [1 x];

  % Calculando saída da camada escondida
  yh = a*tanh(b*xin*w1');

  % Adicionando entrada y0 para a camada de saída (bias)
  xh = [1 yh];

  % Calculando saída da camada de saída
  y = a*tanh(b*xh*w2');
endfunction


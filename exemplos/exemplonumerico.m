clear all

etaSaida=0.7;
etaEscondida=0.5;

x = [1 5 2];
d = [0.4 -0.3];
w = [-0.2  0.1  0.7;
      0.3 -0.7  0.8;
      0.5  0.3 -0.6;
      0.4  0.5 -0.4];
W = [ 0.1  0.3 -0.4  0.7  0.6;
     -0.7  0.8  0.3 -0.4  0.5];

% Saída da camada escondida
yh = tanh(x*w')

% Saída da camada de saída
y = tanh([1 yh]*W')

% Erro
e = d - y

% Gradiente da camada de saída
deltaSaida = e.*(1-y).*(1+y)

% Atualizando pesos da camada de saida
Watual = W + etaSaida*deltaSaida'*[1 yh]

% Matriz de pesos da camada de saida reduzida (sem os pesos relativos ao bias)
t = size(W);
Wred = W(:,2:t(2))

% Gradiente local da camada escondida
deltaEscondida = (1-yh).*(1+yh).*(deltaSaida*Wred)

% Atualizando pesos da camada escondida
watual = w + etaEscondida*deltaEscondida'*x 

y = forward(watual,Watual,[5 2])

erro = d-y


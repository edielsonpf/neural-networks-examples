function y=forward(wji,wkj,x,a,b)
%Entrada:
%wji=> pesos neurônios de entrada
%wkj=> pesos neurônio escondida
%x=> valores de entrada (vetor coluna)
%a=> valor para multiplicar a tanh
%b=> valor para multiplicar o argumento da tanh

% Adicionando entrada x0 (bias)
x=[1;x];

% Calculando saída da camada escondida
vj=wji*x;
yesc=a.*tanh(b.*vj);

% Adicionando entrada y0 para a camada de saída (bias)
yj=[1;yesc];

% Calculando saída da camada de saída
ysaida=a.*tanh(b.*(wkj*yj));
y=ysaida;


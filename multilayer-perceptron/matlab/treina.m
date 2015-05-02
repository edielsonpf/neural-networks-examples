function [w1,w2] = treina(xt,rt,xv,rv,nEntradas,nSaidas,nEscondidas)
%xt: vetor para treinamento
%rt: vetor com resultados para treinamento
%xv: vetor para validação cruzada
%rv: vetor com resultados para validação cruzada


% Passos de aprendizagem
eta1=0.02; % fator de aprendizagem para a camada de entrada (escondida)
eta2=0.01; % fator de aprendizagem para a camada de saida

% Inicializando matrizes de pesos sinapticos
c1 = 0.5; % faixa de variacao dos pesos iniciais da camada escondida
c2 = 0.5; % faixa de variacao dos pesos iniciais da camada de saida
w2=unifrnd(-c2,c2,nSaidas,nEscondidas+1); % camada de saida
w1=unifrnd(-c1,c1,nEscondidas,nEntradas+1); % camada de entrada
delta1=zeros(size(w1)); % camada de entrada
delta2=zeros(size(w2)); % camada de saida
%a = 1.7159;
%b = 2.0/3.0;
a=1;
b=1;

% Numero de locucoes de treinamento e validacao cruzada
Nt = length(rt);
Nv = length(rv);

% Treinando a rede
limiar = 1e-3;
medio(1)=1;
epocas=1;
while (medio(epocas) > limiar)&(epocas<1000) 
   epocas=epocas+1;
   
   for i=1:Nt
      
      % Entrada modificada (insercao do limiar de ativacao)
      y0=[1;xt(:,i)];
      
      % Saida da camada escondida
      y1=a*tanh(b.*(w1*y0));

      % Entrada da camada de saida
      y2=[1;y1];

      % Saida da camada de saida (saida da rede)
      y3=a*tanh(b.*(w2*y2));
      
      % Calculo do gradiente local para a camada de saida
      delta2 = b/a*(rt(i)-y3).*(a-y3).*(a+y3);
         
      % Atualizacao dos pesos da camada de saida
      w2=w2+eta2*delta2*y2';
   
      % Calculo do gradiente local para a camada escondida
      w2r = w2(:,2:nEscondidas+1);
      aux = w2r'*delta2;
      delta1=b/a*(a-y1).*(a+y1).*aux;
   
      % Atualizacao dos pesos da camada escondida
      w1=w1+eta1*delta1*y0';
   end

   % Calculo do erro quadratico medio
   soma = 0;
   for i=1:Nv
	y=forward(w1,w2,xv(:,i),a,b);	
	erro= (rv(i)-y).^2;
	soma = soma + erro;
   end
   medio(epocas) = soma/Nv;
  
end
medio(epocas)
plot(medio(2:length(medio)))
end


function [w1,w2]=treina(xt,dt,xv,dv,nEntradas,nEscondida,nSaidas)

% Passos de aprendizagem
eta1=0.002; % fator de aprendizagem para a camada de entrada (escondida)
eta2=0.001; % fator de aprendizagem para a camada de saida

% Inicializando matrizes de pesos sinapticos
c1 = 0.001; % faixa de variacao dos pesos iniciais da camada escondida
c2 = 0.001; % faixa de variacao dos pesos iniciais da camada de saida
w1=uniform_rnd(-c1,c1,nEscondida,nEntradas+1); % camada de entrada
w2=uniform_rnd(-c2,c2,nSaidas,nEscondida+1); % camada de saida
delta1=zeros(size(w1)); % camada de entrada
delta2=zeros(size(w2)); % camada de saida
a = 1.7159;
b = 2.0/3.0;

% Numero de locucoes de treinamento e validacao cruzada
Nt = length(dt);
Nv = length(dv);

% Treinando a rede
limiar = 1e-4;
medio(1)=1;
epocas=1;
while (medio > limiar)&(epocas<10000) 
   epocas=epocas+1;

   [xt,dt] = embaralha(xt,dt);
   
   for i=1:Nt
      
      % Entrada modificada (insercao do limiar de ativacao)
      y0=[-1;xt(:,i)];
      
      % Saida da camada escondida
      y1=a*tanh(b*w1*y0);

      % Entrada da camada de saida
      y2=[-1;y1];

      % Saida da camada de saida (saida da rede)
      y3=a*tanh(b*w2*y2);
      
      % Calculo do gradiente local para a camada de saida
      delta2 = b/a*(dt(i)-y3).*(a-y3).*(a+y3);
         
      % Atualizacao dos pesos da camada de saida
      w2=w2+eta2*delta2*y2';
   
      % Calculo do gradiente local para a camada escondida
      w2r = w2(:,2:nEscondida+1);
      aux = w2r'*delta2;
      delta1=b/a*(a-y1).*(a+y1).*aux;
   
      % Atualizacao dos pesos da camada escondida
      w1=w1-eta1*delta1*y0';
   end

   % Calculo do erro quadratico medio
   soma = 0;
   for i=1:Nv
      soma = soma + (dv(i)-forward(xv(:,i),w1,w2)).^2;
   end
   medio(epocas) = soma/Nv;
   medio(epocas);
end

plot(medio(2:length(medio)))

end
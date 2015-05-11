function [xemb,demb] = embaralha(x,d);

% Numero de pares estimulo-resposta
nExemplos = length(d);

% Inicializando gerador de numeros aleatorios
rand('state',sum(100*clock));

for i=1:nExemplos
   p1=round((nExemplos-1)*rand+1);
   p2=round((nExemplos-1)*rand+1);
   xaux = x(:,p1);
   daux = d(p1);
   x(:,p1) = x(:,p2);
   d(p1) = d(p2);
   x(:,p2) = xaux;
   d(p2) = daux;
end
xemb = x;
demb = d;



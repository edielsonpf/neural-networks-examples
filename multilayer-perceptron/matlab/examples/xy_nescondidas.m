clear all
% Dados da rede neural
nEntradas = 2; % numero de sinais de entrada
%nEscondidas = 4; % numero de neuronios da camada de entrada
nSaidas = 1; % numero de neuronios da camada de saida

a = 1;
b = 1;


% Gerando pares de treinamento
N = 4000; % numero de pares de treinamento
x=uniform_rnd(0,1,nEntradas,N);
d=zeros(1,N); % saidas desejadas
for i=1:N
	if((x(1,i)<=0.5 & x(2,i)<=0.5)|(x(1,i)>0.5 & x(2,i)>0.5))
		d(i)=-0.5;  
	else
		d(i)=0.5;
	end
end

% Gerando pares de validacao cruzada
Nv = 1000; % numero de pares de treinamento
xv=uniform_rnd(0,1,nEntradas,Nv);
dv=zeros(1,Nv); % saidas desejadas
for i=1:Nv
	if((xv(1,i)<=0.5 & xv(2,i)<=0.5)|(xv(1,i)>0.5 & xv(2,i)>0.5))
		dv(i)=-0.5;  
	else
		dv(i)=0.5;
	end
end

% Gerando pares de teste
Nt = 1000; % numero de pares de treinamento
xt=uniform_rnd(0,1,nEntradas,Nt);
dt=zeros(1,Nt); % saidas desejadas
for i=1:Nt
	if((xt(1,i)<=0.5 & xt(2,i)<=0.5)|(xt(1,i)>0.5 & xt(2,i)>0.5))
		dt(i)=-0.5;  
	else
		dt(i)=0.5;
	end
end

for nEscondidas=1:10
	[w1,w2] = treina(x,d,xv,dv,nEntradas,nSaidas,nEscondidas);
	
	acerto(nEscondidas) = testaxy(xt,dt,Nt,w1,w2);
	% Testando a rede neural
	%for i=1:Nt
   	%	err(i) = (dt(i)-forward(w1,w2,xt(:,i),a,b)).^2;
	%end

	%if(nEscondidas==1)
	%	media = mean(err)
	%	nEscondidas
	%else
	%	if(mean(err)<media)
	%		media=mean(err)
	%		nEscondidas=nEscondidas
	%	else
	%		break;
	%	end
	%end
	%media(nEscondidas)=mean(err);
	%nEscondidas=nEscondidas
	
end
t=1:10;
clf
plot(t,acerto)

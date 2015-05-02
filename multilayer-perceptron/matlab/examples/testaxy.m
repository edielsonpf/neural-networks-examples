function acerto = testaxy(xt,dt,Nt,w1,w2)
	acerto=0;
	cont=1;

	for i=1:Nt
   		y=forward(w1,w2,xt(:,i),1,1);
		
		if(y>0)
			yv=0.5;
		else
			yv=-0.5;
		end

		if(yv==dt(i))
			acerto++;
		else
			indice(cont)=i;
			cont++;
		end
	end

	clf		
	for i=1:cont-1
		axis([-0.5 1.5 -0.5 1.5])
		plot(xt(1,indice(i)),xt(2,indice(i)),"@")
		hold on
	end
	acerto=(acerto/Nt)*100;
	pause

%Definiçao de variaveis
in =[0 0;0 1;1 0;1 1];
d = [0 0 0 1]; %porta AND

ETA = zeros(5,20);
limiar = 0;

    fid = fopen('E:\INATEL\Mestrado\tp511\TP511\MatLab\relatorio.txt','w+');
    fprintf (fid,'------------------RELATORIO DAS EXPERIENCIAS------------\n');
    fprintf (fid,'  W0          W1          W2          ETA     N DE EPOCAS\n');
    fprintf (fid,'----------------------------------------------------------\n');
    fclose(fid);

 for k=1:10
     w1 = rand(1,3)
    eta = 0.1;
    %Teste para ver qual o melhor valor de eta entre 0.1 e 2.0 para obter
    %um menor número de épocas para o sistema.
    for e=1:20

        w = w1;
%        w = [0.3 0.4 0.3];
        ETA(1,k) = w(1);
        ETA(2,k) = w(2);
        ETA(3,k) = w(3);

        epoca = 0;
        erro = 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %************************EPOCA**************************************
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        while(erro>limiar) %enquanto erro maior que limiar entra na estrutura
            %--------------------------TREINAMENTO---------------------------
            for i=1:length(d)
                x = [1 in(i,:)];
                u = x*w';
                if(u>=0)
                    y=1;
                else
                    y=0;
                end;
                erro = d(i)-y;
                for j=1:length(w)
                    w(j)=w(j) + eta*x(j)*erro;
                end;
            end;
            %-----------------------------------------------------------------
            %///////////////////////////TESTE////////////////////////////////
            erro = 0;
            for i=1:length(d)
                x = [1 in(i,:)];
                u = x*w';
                if(u>=0)
                    y=1;
                else
                    y=0;
                end;
                erro = abs(d(i)-y) + erro;
            end;
            %/////////////////////////////////////////////////////////////////
            epoca = epoca +1; %conta quantas épocas foram necessárias
        end;
        %********************************************************************

    fid = fopen('E:\INATEL\Mestrado\tp511\TP511\MatLab\relatorio.txt','a+');
    if (epoca < 2)
        fprintf (fid,'%6.3f %11.3f %11.3f %11.3f* %11.3f* \n',w(1),w(2),w(3),eta,epoca);
    else
        fprintf (fid,'%6.3f %11.3f %11.3f %11.3f %11.3f \n',w(1),w(2),w(3),eta,epoca);
    end;
           
    fclose(fid);
        
        ETA(4,e) = eta;
        ETA(5,e) = epoca;
        eta = eta + 0.1;

    end;

    fid = fopen('E:\INATEL\Mestrado\tp511\TP511\MatLab\relatorio.txt','a+');
    fprintf (fid,'----------------------------------------------------------\n');
    fprintf (fid,'------------------OUTROS VALORES DE W---------------------\n');
    fprintf (fid,'  W0          W1          W2          ETA     N DE EPOCAS\n');
    fprintf (fid,'----------------------------------------------------------\n');
    fclose(fid);
        
end;



      

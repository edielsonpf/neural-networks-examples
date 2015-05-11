function erro = valida(w1,w2,a,b,xv,dv)

  t = size(xv);
  nValidacao = t(2);

  erro = 0;
  for i=1:nValidacao
    y = forward(w1,w2,a,b,xv(i,:));       % calculando saida da rede
    e = dv(i,:)-y;
    erro = erro + e*e';
  endfor 
  erro = erro/nValidacao;  
endfunction


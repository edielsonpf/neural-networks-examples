// Classifica a doença de acordo com as caracteristicas colhidas no exame de
// sangue
// Parâmetros de entrada:
// entradas: Entradas do conjunto de treinamento
// pesos: Pesos iniciais da rede
// Parâmetros de saída:
// classes: Classificação das amostras passadas como entrada
function classes = perceptron_operation(w, x)
    // Número de amostras
    [Input_Size,Samples_Size]=size(x);
    // Inicializa o vetor de classes (tipos classificados)
    classes = zeros(Samples_Size, 1);
    for k = 1 : Samples_Size
        v=w'*x(:,k);
        //y = sign(v);
        y = tanh(v);
        if (y <= 0)
            classes(k) = -1;
        else
            classes(k) = 1;
        end
        //classes(k)=y;
    end
    disp(classes);
endfunction

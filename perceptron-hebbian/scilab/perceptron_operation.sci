// Classifica a doença de acordo com as caracteristicas colhidas no exame de
// sangue
// Parâmetros de entrada:
// entradas: Entradas do conjunto de treinamento
// pesos: Pesos iniciais da rede
// Parâmetros de saída:
// classes: Classificação das amostras passadas como entrada
function classes = perceptron_operation(pesos, entradas)
    // Número de amostras
    num_amostras = size(entradas, 1);
    // Inicializa o vetor de classes (tipos classificados)
    classes = zeros(num_amostras, 1);
    for k = 1 : num_amostras
        u = entradas(k,:) * pesos;
        y = sign(u);
        if (y == -1)
            classes(k) = -1;
        else
            classes(k) = 1;
        end
    end
    disp(classes);
endfunction

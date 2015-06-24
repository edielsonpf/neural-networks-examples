load('ANN.sod','W','np','n1');
//load('ANN_np_5_n1_10.sod','W','np','n1');
disp(W);
disp(np);
disp(n1);

//===============Running===========
TestData=[
0.0195
0.4366
0.0924
0.7984
0.0077
0.4173
0.0062
0.3387
0.1886
0.7418
0.3138
0.4466
0.0835
0.1930
0.3807
0.5438
0.5897
0.3536
0.2210
0.0631
0.4499
0.2564
0.7642
0.1411
0.3626
];
//Preparing testing data
[r,c]=size(TestData);
xt=[];
dt=[];
nd=1;
for t=np+1:1:r
    for i=1:np
        xt(i,t-np)=TestData(t-i);
    end
    dt(1,t-np)=TestData(t);
    nd=nd+1;
end
disp(xt);

NeuralNetwork = [np n1 1];
y = ann_FF_run(xt,NeuralNetwork,W);
disp(y);
disp(dt);
Erro = eqm(y,dt);
disp(Erro);





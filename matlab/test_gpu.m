X = csvread('DATA_4_MT.dat');
Y = csvread('DATA_2_MT.dat');
Z = csvread('DATA_3_MT.dat');

tic;
m_s = mean(X)';
m_vi = mean(Y)';
m_ve = mean(Z)';

m = (m_s + m_vi + m_ve)./3;

c_s = cov(X);
c_vi = cov(Y);
c_ve = cov(Z);

sw = c_s + c_vi + c_ve;

%numero campioni per ogni classe
Ns = size(X,2);
Nvi = size(Y,2);
Nve = size(Z,2);


SBs =  (m_s - m)*(m_s - m)';
SBvi =  (m_vi - m)*(m_vi - m)';
SBve =  (m_ve - m)*(m_ve - m)';

%Between totale
SB = (SBs + SBvi + SBve)/8;

%LDAm
invSw = inv(sw);
invSw_by_SB = invSw * SB;

%D autovalori e V autovettori
[V,D] = eig(invSw_by_SB);

W1 = V(:,1);
W2 = V(:,2);


XN = X*[W1,W2];
YN = Y*[W1,W2];
ZN = Z*[W1,W2];     

toc;
f1 = figure('Name','Measured Data');
f2 = figure('Name','Measured Data');

figure(f1);
plot(X(:,1),X(:,2),'*');
hold on;
plot(Y(:,1),Y(:,2),'*');
hold on;
plot(Z(:,1),Z(:,2),'*');

figure(f2);
plot(XN(:,1),XN(:,2),'*');
hold on;
plot(YN(:,1),YN(:,2),'*');
hold on;
plot(ZN(:,1),ZN(:,2),'*');

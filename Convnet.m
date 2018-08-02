%% MNIST Handwritten Digit Classification using Convolution Neural network

clear;
clc;
% Loading Data

% load('mnist_all.mat');
% x=[train0,ones(length(train0),1).*0;train1,ones(length(train1),1).*1;train2,ones(length(train2),1).*2;train3,ones(length(train3),1).*3;train4,ones(length(train4),1).*4;train5,ones(length(train5),1).*5;train6,ones(length(train6),1).*6;train7,ones(length(train7),1).*7;train8,ones(length(train8),1).*8;train9,ones(length(train9),1).*9];
% y=x(:,785);
% y=double(y);
% x(:,785)=[];
% m=length(x);
% xtest=[test0,ones(length(test0),1).*0;test1,ones(length(test1),1).*1;test2,ones(length(test2),1).*2;test3,ones(length(test3),1).*3;test4,ones(length(test4),1).*4;test5,ones(length(test5),1).*5;test6,ones(length(test6),1).*6;test7,ones(length(test7),1).*7;test8,ones(length(test8),1).*8;test9,ones(length(test9),1).*9];
% ytest=xtest(:,785);
% ytest=double(ytest);
% xtest(:,785)=[];
% mtemp=length(xtest);
% 
% x=double(x');
% xtest=double(xtest');
% y=double(y);
% ytest=double(ytest);

% save('x.mat');
% save('xtest.mat');
% save('y.mat');
% save('ytest.mat');

load('x.mat');
load('xtest.mat');
load('y.mat');
load('ytest.mat');
m=60000;
mtemp=10000;

Y=zeros(10,m);
q=1;
while q<=m
    Y(y(q)+1,q)=1;
    q=q+1;
end



% Initialization of Parameters/Hyperparameters

F1=rand(5,5,3);
w1=rand(100,432);
w2=rand(10,100);
%B1=rand(1,1,3);
b1=rand(100,1);
b2=rand(10,1);

alpha=0.2;
noofiter=2;

dw2=rand(10,100);
dw1=rand(100,432);
db1=zeros(100,1);
db2=zeros(10,1);
dF1=rand(size(F1));




% Non-Vectorized Implementation of Gradient Descent

%i=1;
for i=1:noofiter
    i
    j=1;
    while j<=m
        
        % Forward Propagation
        
        % Convolution-2D
        
        xtemp=x(:,j);
        xtemp=reshape(xtemp,[28,28]);
        C=convforward(xtemp,F1,1,0);
        C1=relu(C);
        
        % Pooling (Max-Pooling)
        
        P=maxpooling(C1,2,2,0);
        a0=reshape(P,[432,1]);
        
        % Fully connected network
        
        z1=w1*a0 + b1;
        a1=sigmoid(z1);
        z2=w2*a1+b2;
        h=sigmoid(z2);
        
        % Backpropagation
        
        dz2=(h-Y(:,j));
        dw2=dw2+(dz2*a1');
        db2=db2+dz2;
        dz1=(w2'*dz2).*(a1.*(1-a1));
        dw1=dw1+(dz1*a0');
        db1=db1+dz1;
        
        da0=w1'*dz1;
        dP=reshape(da0,[12,12,3]);
        dC=poolbackward(dP,P,2,2);
        dC(C<0)=0;
        [dX,dF]=convbackward(dC,xtemp,F1);
        dF1=dF1+dF;
        
        j=j+1;
    end
    F1=F1-(alpha/m)*dF1;
    w1=w1-(alpha/m)*dw1;
    w2=w2-(alpha/m)*dw2;
    b1=b1-(alpha/m)*db1;
    b2=b2-(alpha/m)*db2;
    
    %i=i+1
end

% Testing

%i=1;
parfor i=1:mtemp
    
    xtemp=xtest(:,i);
    xtemp=reshape(xtemp,[28,28]);
    Ct=convforward(xtemp,F1,1,0);
    C1t=relu(Ct);
        
    % Pooling (Max-Pooling)
    
    Pt=maxpooling(C1t,2,2,0);
    a0t=reshape(Pt,[432,1]);
    
    % Fully connected network
    
    z1t=w1*a0t + b1;
    a1t=sigmoid(z1t);
    z2t=w2*a1t+b2;
    ht=sigmoid(z2t);
    [a,b]=max(ht);
    yhyp(i)=b;
    
    %i=i+1;
end

yhyp=yhyp-1;
yhyp=yhyp';
error=abs(ytest-yhyp);
error=sum(error~=0);

accuracy=(1-(error/mtemp))*100
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


function [C] = convforward(X,F,s,p)

% X:Input to the convolution layer ( Size:h x w x 1 )
% W:Filters used ( Size:fxfxnf )
% H:Output of the convolution layer( Size:nh x nw x nf )
% cache:(X,W) - Will be useful while backpropagation
% hyperparam:[stride,padding]

% Note: Add bias and apply relu in the main function

[f,f,nf]=size(F);

[h,w,ch]=size(X);


%i=1;
for i=1:nf
    %j=1;
    for j=1:((h-f+2*p)/s)+1
        %k=1;
        for k=1:((w-f+2*p)/s)+1
            xslide=X(j:j+f-1,k:k+f-1);
            d=(xslide.*F(:,:,i));
            C(j,k,i)=sum(sum(d));
            %k=k+1;
        end
        %j=j+1;
    end
    %i=i+1;
end









end

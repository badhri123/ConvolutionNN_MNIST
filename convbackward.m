function [dX,dF] = convbackward(dH,X,F)

[f,f,nf]=size(F);
[dh,dw,nf]=size(dH);
[h,w,ch]=size(X);
dF=zeros(f,f,nf);
dX=zeros(h,w,ch);

%i=1;
for i=1:nf
    %j=1;
    for j=1:dh
        %k=1;
        for k=1:dw
            xslide=X(j:j+f-1,k:k+f-1);
            dF(:,:,i)=dF(:,:,i)+(xslide.*dH(j,k,i));
            dX(j:j+f-1,k:k+f-1,i)=dX(j:j+f-1,k:k+f-1,i)+(F(:,:,i).*dH(j,k,i));
            %k=k+1;
        end
        %j=j+1;
    end
    %i=i+1;
end



end


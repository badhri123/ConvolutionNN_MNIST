function [P] = maxpooling(X,f,s,p)

% By default ,Max-pooling is done

[h,w,ch]=size(X);
%i=1;
for i=1:ch
    %j=1;
    for j=1:((h+2*p-f)/s)+1
        %k=1;
        for k=1:((w+2*p-f)/s)+1
            P(j,k,i)=max(max((X(j:j+f-1,k:k+f-1))));
            %k=k+1;
        end
        %j=j+1;
    end
    %i=i+1;
end




end


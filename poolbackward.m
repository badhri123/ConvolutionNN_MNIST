function [dH_prev] = poolbackward(dH,H_prev,s,f)

[nh,nw,nch]=size(dH);
dH_prev=zeros(size(H_prev));
%i=1;
for i=1:nch
    %j=1;
    for j=1:s:nh
        %k=1;
        for k=1:s:nw
            wind=H_prev(j:j+f-1,k:k+f-1,i)==max(max(H_prev(j:j+f-1,k:k+f-1,i)));
            dH_prev(j:j+f-1,k:k+f-1,i)=wind.*dH(j,k,i);
            %k=k+s;
        end
        %j=j+s;
    end
    %i=i+1;
end

end


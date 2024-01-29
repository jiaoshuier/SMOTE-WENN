function discard = WENN(instances,k,IR,AttVector)
% ENN: Edited Nearest Neighbor
% reference: Wilson D R, Martinez T R. Reduction techniques for instance-based
% learning algorithms[J]. Machine learning, 2000, 38(3): 257-286.

attribute = VDM(instances,AttVector);

data = instances(:,1:end-1);
[h,l] = size(data);
target = instances(:,end);
npos = 1/(1+IR);
nneg = IR/(1+IR);
discard = [];% the index of negtive examples in data for removing

a = std(data,0,1);
for i = 1:h
    d = [];
    for j = find(AttVector==0)
        d = [d (repmat(data(i,j),h,1)-data(:,j))/(4*a(j))];
    end
    for j = find(AttVector==1)
        id1 = Locate(attribute(j).values,data(i,j));
        id2 = Locate(attribute(j).values,data(:,j));
        d = [d attribute(j).VDM(id2,id1)];
    end
    d = sum(d.^2,2);  
    d = sqrt(d);
%       
     d(target==1,:) = d(target==1,:)*exp(npos^l);   %注释掉这两行是ENN
     d(target==0,:) = d(target==0,:)*exp(nneg^l);
    
    d(i) = Inf;
    if(k < log2(h))
        min_id = [];
        for j = 1:k
            [~,id] = min(d);
            d(id) = Inf;
            min_id=[min_id id];% sort>=O(n*logn),so we take min: O(n).total time:O(k*n)
        end
    else
        [tmp,id] = sort(d);
        min_id = id(1:k);
    end
    if sum(target(min_id)~=target(i))>=(k+1)/2
       discard = [discard;i];
    end
     
end

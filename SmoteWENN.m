function curtrain_instances = SmoteWENN(train_instances,neighbor,IR,categorical)
% Usage:
%   curtrain_instances: instance matrix after using SmoteWENN. 
%                
%  train_instances   : orginal training samples. No.instances * (No.attribute+class)     
%  neighbor   : neighbor number used in WENN.
%  IR: imbalance ratio 
%  caterorical: to index which attribute is nominal.

%------prepare for distance function------
pos_data = train_instances(train_instances(:,end)==1,1:end-1);
AttVector = zeros(1,size(pos_data,2));
AttVector(categorical) = 1;
attribute = VDM(train_instances,AttVector);

%------SMOTE----------
negnum = sum(train_instances(:,end)==0);
N = negnum-size(pos_data,1); % number of new samples to generate
k = 5;

pos_sample = SMOTE(pos_data,N,k,AttVector,attribute);
pos_ins = [pos_sample ones(N,1)];
bal_instances = [train_instances;pos_ins];

 %-----WENN--------
discard = WENN(bal_instances,neighbor,IR,AttVector);
bal_instances(discard,:) = [];
curtrain_instances = bal_instances;
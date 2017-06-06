function vectors=convert_labels_to_vectors(labels, outputSize)

% labels=transpose(labels);
% 
% %labels(labels==0)=10;
% labels=labels+1;
% 
% labels2=[transpose(1:size(labels,1)),labels];
% 
% vectors=zeros(size(labels,1),10);
% 
% selectElement={labels2(:,1),labels2(:,2)};
% idx=sub2ind(size(vectors), selectElement{:});
% vectors(idx)=true;
% vectors=transpose(vectors);



labels=transpose(labels);

labels=labels+1;

labels2=[transpose(1:size(labels,2)),labels'];

vectors=zeros(size(labels,2),outputSize);

selectElement={labels2(:,1),labels2(:,2)};
idx=sub2ind(size(vectors), selectElement{:});
vectors(idx)=true;
vectors=transpose(vectors);

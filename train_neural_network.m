function neuralNetwork = train_neural_network(structure, miniBatchSize, nEpochs, eta, lambda, mu, noImprovementIn, trainImages, trainLabels, validationImages, validationLabels)

%% Initialise Neural Network

neuralNetwork.sizes=structure; %number of neurons in each layer;
neuralNetwork.layers=size(neuralNetwork.sizes,2);
for j=1:size(neuralNetwork.sizes,2)
    
    if j>1
        neuralNetwork.biases{j-1}=randn(neuralNetwork.sizes(1,j),1);
    end
    
    if j<size(neuralNetwork.sizes,2)
        initialSD=1./sqrt(neuralNetwork.sizes(1,j));
        neuralNetwork.weights{j}=initialSD.*randn(neuralNetwork.sizes(1,j+1),neuralNetwork.sizes(1,j));
    end
end

n=size(trainImages,2);
nBatches=n/miniBatchSize;
for k=1:size(neuralNetwork.weights,2)
    wVel{k}=zeros(size(neuralNetwork.weights{k}));
end
for k=1:size(neuralNetwork.weights,2)
    bVel{k}=zeros(size(neuralNetwork.weights{k},1),1);
end
accuracyMax=0;
maxCount=0;
for epoch=1:nEpochs
    
    %% Set mini-batches
    % shuffle training data    
    idx=randperm(size(trainImages,2));
    imagesShuffle=trainImages(:,idx);
    labelsShuffle=trainLabels(idx);
    
    % put shuffled training data into batches
    imagesRow=size(imagesShuffle,1);
    imagesCol=zeros(1,nBatches)+miniBatchSize;
    batchesImages=mat2cell(imagesShuffle, imagesRow, imagesCol);
    batchesLabels=mat2cell(labelsShuffle, imagesCol,1);
    
    %% Update mini-batch    
    for bc=1:size(batchesImages,2)
        input=batchesImages{bc};
        m=size(input,2);
        desiredOutput=batchesLabels{bc};
        desiredOutput=desiredOutput;
        desiredOutputVectors=convert_labels_to_vectors(desiredOutput, neuralNetwork.sizes(end));
              
        for ioc=1:size(input,2)
            

            [nabla_b, nabla_w]=backpropagation(neuralNetwork, input(:,ioc), desiredOutputVectors(:,ioc));
            
            if ioc<2
                for nc=1:size(nabla_w,2)
                    sum_nb{nc}=zeros(size(nabla_b{nc},1),size(nabla_b{nc},2));
                    sum_nw{nc}=zeros(size(nabla_w{nc},1),size(nabla_w{nc},2));
                end
            end
            
            for sc=1:size(nabla_w,2)                
                sum_nb{sc}=sum_nb{sc}+nabla_b{sc};
                sum_nw{sc}=sum_nw{sc}+nabla_w{sc};                
            end
        end
        

        for uc=1:size(neuralNetwork.weights,2)            
            
            gradCw0=sum_nw{uc}./m;
            gradCw=gradCw0+lambda.*neuralNetwork.weights{uc}./n;
            wVel{uc}=mu.*wVel{uc}-eta.*gradCw;
            neuralNetwork.weights{uc}=neuralNetwork.weights{uc}+wVel{uc};
            %neuralNetwork.weights{uc}=neuralNetwork.weights{uc}+mu.*wVel{uc}; %nesterov part
            
            %neuralNetwork.weights{uc}=(1-eta*(lambda/n))*neuralNetwork.weights{uc}-(eta/m).*sum_nw{uc};
            
            
            gradCb=sum_nb{uc}./m;
            bVel{uc}=mu.*bVel{uc}-eta.*gradCb;
            neuralNetwork.biases{uc}=neuralNetwork.biases{uc}+bVel{uc};
            %neuralNetwork.biases{uc}=neuralNetwork.biases{uc}+mu.*bVel{uc};
            
            %neuralNetwork.biases{uc}=neuralNetwork.biases{uc}-(eta/m).*sum_nb{uc};
        end
    end
    
    
    
    %% find total number of images predicted correctly
    
    for ic=1:size(validationImages,2)
        
        input=validationImages(:,ic);
        inputLabel=validationLabels(ic);
        activation=input;
        for lc=1:size(neuralNetwork.weights,2)
            activation=sigmoid(neuralNetwork.weights{lc}*activation+neuralNetwork.biases{lc});
        end
        [maxValue, idx]=max(activation);
        result=idx-1;
        
        results(ic)=result==inputLabel;
        
    end
    
    totalCorrect(epoch,:)=[sum(results),size(validationImages,2)];
    
    fprintf('Epoch %i: %0.3f\n', epoch, 100*sum(results)/size(validationImages,2));
    
    %% No improvement in n
    if totalCorrect(epoch,1)>accuracyMax
        accuracyMax=totalCorrect(epoch,1);
        maxCount=0;
    else
        maxCount=maxCount+1;
    end
    
    if maxCount>noImprovementIn
        break
    end
    
    amh(epoch,:)=accuracyMax;
    
end





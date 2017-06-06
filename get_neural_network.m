clear all; clc;
rng(0);

%% Set data

n=50000;

trainImagesRaw=loadMNISTImages('train-images.idx3-ubyte');
trainImages = reshape(trainImagesRaw, size(trainImagesRaw, 1) * size(trainImagesRaw, 2), size(trainImagesRaw, 3));
trainLabels=loadMNISTLabels('train-labels.idx1-ubyte');

testImagesRaw=loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImagesRaw, size(testImagesRaw, 1) * size(testImagesRaw, 2), size(testImagesRaw, 3));
testLabels=loadMNISTLabels('t10k-labels-idx1-ubyte');

validationImagesRaw=trainImagesRaw(:,:,n+1:end);
validationImages=trainImages(:,n+1:end);
validationLabels=trainLabels(n+1:end,1);

trainImagesRaw=trainImagesRaw(:,:,1:n);
trainImages=trainImages(:,1:n);
trainLabels=trainLabels(1:n,1);

% trainImagesRaw=importdata('test_alteredAll.mat');
% trainImages = reshape(trainImagesRaw, size(trainImagesRaw, 1) * size(trainImagesRaw, 2), size(trainImagesRaw, 3));
% trainLabels=loadMNISTLabels('train-labels.idx1-ubyte');
% n=size(trainImagesRaw,3);


%% Set hyper-parameters
Nh=100;
miniBatchSize=10;
nEpochs=75;
eta=0.1;%0.5; %learning rate
lambda=5;%0.1; %regularization
mu=0.3; %momentum coefficient
noImprovementIn=200;

Ni=size(trainImages,1);
No=size(unique(trainLabels),1);

%% Train network
structure=[Ni, Nh, No];
neuralNetwork=train_neural_network(structure, miniBatchSize, nEpochs, eta, lambda,...
    mu, noImprovementIn, trainImages, trainLabels, validationImages, validationLabels);


% %% show a sample of images and their respective prediction
% select=[1,2,4,6];
% for ic=1:4%size(neuralNetwork.weights,2)
%     input=imagesShuffle(:,select(ic));
%     inputLabel=labelsShuffle(select(ic));
%     activation=input;
%     for lc=1:size(neuralNetwork.weights,2)
%         activation=sigmoid(neuralNetwork.weights{lc}*activation+neuralNetwork.biases{lc});
%     end
%     [maxValue, idx]=max(activation);
%     result=idx-1;
%     
%     
%     
%     I=reshape(input,28,28);
%     I2=imresize(I,3);
%     %imshow(I2);
%     
%     figure(1);
%     subplot(2,2,ic)
%     imshow(I2);
%     %title(['Input:', num2str(inputLabel),', Output:', num2str(result)]);
%     title(['Neural Network Output: ', num2str(result)]);
%     
%     %pause(1);
%     
% end
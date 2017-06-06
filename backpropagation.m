function [nabla_b, nabla_w] = backpropagation(neuralNetwork, input, desiredOutput)

%% changes
% changed cost function to cross entropy

%% initialise 
for i=1:neuralNetwork.layers
    
    if i>1
        nabla_b{i-1}=zeros(size(neuralNetwork.biases{i-1}));
    end
    
    if i<neuralNetwork.layers
        nabla_w{i}=zeros(size(neuralNetwork.weights{i}));
    end
end

%% feedforward
activation=input;
activations{1}=activation;

for j=1:neuralNetwork.layers-1

    
    weight=neuralNetwork.weights{j};
    bias=neuralNetwork.biases{j};
    zs{j}=weight*activation+bias;
    
    sigmoidz=1./(1+exp(-zs{j}));
    activation=sigmoidz;
    activations{j+1}=activation;
    
    
end

%% backward pass

gradC=activations{end}-desiredOutput;
delta=gradC;

nabla_b{end}=delta;
nabla_w{end}=delta*transpose(activations{end-1});


for l=neuralNetwork.layers-1:-1:2
    
   z=zs{l-1};
   weight=neuralNetwork.weights{l};
   sp=(1./((1+exp(-z)))).*(1-1./((1+exp(-z))));
   
   delta=transpose(weight)*delta.*sp;
   nabla_b{l-1}=delta;
   nabla_w{l-1}=delta*transpose(activations{l-1});
   
    
end





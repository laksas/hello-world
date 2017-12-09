 require 'nn'
 require 'gnuplot'
 require 'tools' 
 require 'optim'
 require 'torch'
 require 'nn'
 require 'dp'
 require 'xlua'
 

 -- 1: Creation de la base  de données

classes = {'1','2','3','4','5','6','7','8','9','0'}

--  matrix de confusion des classes
confusion = optim.ConfusionMatrix(classes)

print '==> downloading dataset'
tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'


print '==> loading dataset'
print '==> downloading dataset'
tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'


print '==> loading dataset'
 
 
 -- 2 : creation du modele
 
local model = nn.Sequential()
model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 50))
model:add(nn.ReLU())
model:add(nn.Linear(50,10))
model:add(nn.ReLU())
model:add(nn.LogSoftMax())
local criterion=nn.ClassNLLCriterion() 
params, grad = model:getParameters()
 
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- crée mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load simple
         local input = trainData.data[shuffle[i]]:double()
         local target = trainData.labels[shuffle[i]]
         table.insert(inputs, input)
         table.insert(targets, target)
      end
   end
end 


 -- 3 : Creation de la fonction de calcul du gradient
feval = function(params_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if params ~= params_new then
      params:copy(params_new)
   end
   grad:zero()

   -- select a new training sample   
   for i =1, #inputs do         --80
   local ouput= model : forward (inputs[i])
   local err =  criterion:forward (ouput, targets[i])
   -- exe retropropagation du model
   local gradouput = criterion :backward(ouput, targets[i])
   model : backward(inputs[i], gradouput)
   confusion:add(output, targets[i])
   end 
   return err, grad

end

optim_params = {
   learningRate = 1e-3,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0
}

 local maxEpoch=10000
 local all_errs={}
 
 -- 4 : Boucle d'apprentissage
 
 for iteration=1,maxEpoch do
  ------ Mise à jour des paramètres du modèle
      ------ Evaluation de la loss moyenne 
        ------ Evaluation de la loss moyenne 
    local err =0
    for j=1,#inputs  do
      local input=inputs[j]
      local target=ouputs[j]
      local out=model:forward(inputs)
      err = err +criterion:forward(output,targets)

    end
    err = err/#inputs
    all_errs[iteration]=err    
      
    -- apprentissage
    err=0
    for j=1,#inputs do
      _,fs=optim.sgd(feval,params,optim_params)
      err=err+fs[1]
    end
    err = err/#inputs 
   -- print confusion matrix
   print(confusion)
  confusion:zero()
  local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
--epoch = epoch + 1
return model

    
  
  -- plot de la frontiere
 -- plot(inputs,targets,model,100)  
  
  -- plot du err
  gnuplot.plot(torch.Tensor(all_errs))
end
 

 

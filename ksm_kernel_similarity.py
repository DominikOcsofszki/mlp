






#Kernel used: Gaussian Kernel with Bandwith=100
##(Eq. 7 - Understanding Counterfactual Generation...)
##Kernel Similarity Measure:
def KSM() :
    return'ksm'

#TODO:
#KSM
#L1
#L2
#Ls

kernel = 'Gaussian kernel'
bandwith = 100
batch_size = 128
#should be nearly proportional of each class
lr = 1e-4 #10^-4
#after 10 epochs, no significally loss change
lr = lr/10
#After 20 epochs no significant change => stopp training
optim = 'Adam'
#Run 5 times each experiment

#SELECT class 4,9
class_nr = (4,9)
x = '400 nodes first layer'
layer1 = 400
layer2 = 20

activation = 'relu'
variance_parameter_beta = 0.5
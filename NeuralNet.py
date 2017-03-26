import numpy as np

class BackPropagationNetwork:
    """A back-propagation network"""

    def __init__(self, layerSize, layerFunctions=None):# layerSize is a tuple of layers
        """Initialize the network"""
        
        self.layerCount = 0
        self.shape = None
        self.weights = []   # weights assigned to a layer are the weights that precede it
                            # so the weights that feed into the layer are the ones which are assigned to it
        
        # Layer info
        self.layerCount = len(layerSize) - 1 # input layer is only a placeholder kinda thing for inputs. So numLayer is 1 less than that
        self.shape = layerSize
        
        # Input Outpur data from last Run
        self._layerInput = []
        self._layerOutput = []

        # layerSize[:-1] All but the last one
        # layerSize[1:] from the first one

        # Create the weight arrays
        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):#(l1,l2) become for say layersize = (2,3,4) : [(2,3),(3,4)]
            self.weights.append(np.random.normal(scale=0.1, size = (l2, l1+1)))# +1 for the bias node
                                                                                # Inputs are 3x4 so weights will be 4xp

def nonlin(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def makeRandomArray(rows, columns):
    '''makes a random array of dimensions nxm'''
    arr = 2*np.random.random((rows,columns)) - 1
    return arr

if __name__ == "__main__":
    n_hiddenlayers = 1
    n_hiddenlayer_neurons = 3
    n_datapoints  = 10
    n_features = 4


    InputData = makeRandomArray(n_datapoints, n_features) # Get Data here!
    OutputData = makeRandomArray(n_datapoints,1) # Get Data here!


    bpn = BackPropagationNetwork((n_features,n_hiddenlayer_neurons,1))
    weights = bpn.weights

    syn0,syn1 = weights[0],weights[1]

    # print "INPUT DATA:    \n",InputData.shape,"\n",InputData

    # print "Syn 0 :    \n",syn0.shape,"\n",syn0

    # print "Syn 1 :    \n",syn1.shape,"\n",syn1 
    
    for j in xrange(60000):
        l0 = InputData
        l0 = np.vstack([l0.T, np.ones(n_datapoints)]).T

        l1 = nonlin(np.dot(l0, syn0.T))
        l1 = np.vstack((l1.T, np.ones(n_datapoints))).T
        l2 = nonlin(np.dot(l1, syn1.T))

        l2_error = OutputData - l2

        if(j % 100) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
            print("Error: " + str(np.mean(np.abs(l2_error))))

        l2_delta = l2_error*nonlin(l2, deriv=True)
        l1_error = l2_delta.dot(syn1)
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn1_delta = l1.T.dot(l2_delta).T
        syn1 += syn1_delta

        l1_delta = l1_delta.T
        l1_delta = np.delete(l1_delta, -1, 0).T
        syn0_delta = l0.T.dot(l1_delta).T
        syn0 += syn0_delta

        # print syn0_delta,syn0_delta.shape


        # syn1 = syn1 + 
        # syn1 = syn1.T +  l1.T.dot(l2_delta)
        # syn0 = l0.T.dot(l1_delta)
        


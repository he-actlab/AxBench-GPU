
def printHeader(af, steep, weights, labels):
    f = open('npuKernelGenerated.h', 'w+')
    f.write('#ifndef __NPUKERNEL_H_\n#define __NPUKERNEL_H_\n\n#include <cuda_runtime_api.h>\n#include <stdio.h>\n#include <math.h>\n\n#ifdef NPU\n__device__ unsigned char\nComputeSobel(unsigned char ul_in, // upper left\n\t\t\t unsigned char um_in, // upper middle\n\t\t\t unsigned char ur_in, // upper right\n\t\t\t unsigned char ml_in, // middle left\n\t\t\t unsigned char mm_in, // middle (unused)\n\t\t\t unsigned char mr_in, // middle right\n\t\t\t unsigned char ll_in, // lower left\n\t\t\t unsigned char lm_in, // lower middle\n\t\t\t unsigned char lr_in, // lower right\n\t\t\t float fScale);\n\n__device__ float af0(float, float);\n__device__ float af1(float, float);\n__device__ float af2(float, float);\n__device__ float af3(float, float);\n__device__ float af4(float, float);\n__device__ float af5(float, float);\n__device__ float af6(float, float);\n__device__ float af7(float, float);\n__device__ float af8(float, float);\n__device__ float af9(float, float);\n__device__ float af10(float, float);\n__device__ float af11(float, float);\n__device__ float af12(float, float);\n__device__ float af13(float, float);\n__device__ float af14(float, float);\n__device__ float af15(float, float);\n__device__ float af16(float, float);\n__device__ float af17(float, float);\n\n#define BIAS 1\n\n// Neuron nomenclature: <Layer>_<Row>, both zero indexed\n\n// Activation function of each neuron\n')

    i = 0
    for neuron in labels:
        f.write('#define af__' + str(neuron) + '(s) af' + str(af[i]) + '(s)\n')
        i += 1 

    f.write("\n// Activation steepness of each neuron\n")
    i = 0
    for neuron in labels:
        f.write('#define as__' + str(neuron) + ' ' + str(steep[i]) + '\n')
        i += 1

    f.write('\n// Weights of each connection\n')
    for w in weights:
        f.write('#define w__' + w + ' ' + str(weights[w]) + '\n')

    f.close()


def printKernel(numLayers, labels, layers):
    f = open('npuKernelGenerated.cu', 'w+')
    f.write('#include "npuKernel.h"\n\n// Define various activation functions from fann_activationfunc_enum\n// http://leenissen.dk/fann/html/files/fann_data-h.html\n// Date: Nov 13, 2014\n__device__ float af0(float sumIn, float steepness) {\n\treturn sumIn * steepness;\n}\n\n//__device__ float af1(float sumIn, float steepness) {\t\n//}\n\n//__device__ float af2(float sumIn, float steepness) {\n//}\n\n__device__ float af3(float sumIn, float steepness) {\n\treturn ( 1.0f / (1 + exp(-2 * steepness * sumIn)) );\n}\n\n//__device__ float af4(float sumIn, float steepness) {\n//}\n\n__device__ float af5(float sumIn, float steepness) {\n\treturn ( 2.0f / (1 + exp(-2 * steepness * sumIn)) - 1 );\n}\n\n//__device__ float af6(float sumIn, float steepness) {\n//}\n\n__device__ float af7(float sumIn, float steepness) {\n\treturn ( exp(-sumIn*steepness*sumIn*steepness) );\n}\n\n__device__ float af8(float sumIn, float steepness) {\n\treturn ( 2*exp(-sumIn*steepness*sumIn*steepness) - 1 );\n}\n\n__device__ float af9(float sumIn, float steepness) {\n\treturn ( 0.5*sumIn*steepness/(1 + fabs(sumIn*steepness)) + 0.5 );\n}\n\n__device__ float af10(float sumIn, float steepness) {\n\treturn ( sumIn*steepness/(1 + fabs(sumIn*steepness)) );\n}\n\n__device__ float af11(float sumIn, float steepness) {\n\treturn ( sumIn * steepness );\n}\n\n__device__ float af12(float sumIn, float steepness) {\n\treturn ( sumIn * steepness );\n}\n\n__device__ float af13(float sumIn, float steepness) {\n\treturn ( sin(sumIn*steepness) );\n}\n\n__device__ float af14(float sumIn, float steepness) {\n\treturn ( cos(sumIn*steepness) );\n}\n\n__device__ float af15(float sumIn, float steepness) {\n\treturn ( sin(sumIn*steepness)*0.5 + 0.5 );\n}\n\n__device__ float af16(float sumIn, float steepness) {\n\treturn ( cos(sumIn*steepness)*0.5 + 0.5 );\n}\n\n#ifdef NPU\n__device__ unsigned char\nComputeSobel(unsigned char ul_in, // upper left\n\t\t\t unsigned char um_in, // upper middle\n\t\t\t unsigned char ur_in, // upper right\n\t\t\t unsigned char ml_in, // middle left\n\t\t\t unsigned char mm_in, // middle (unused)\n\t\t\t unsigned char mr_in, // middle right\n\t\t\t unsigned char ll_in, // lower left\n\t\t\t unsigned char lm_in, // lower middle\n\t\t\t unsigned char lr_in, // lower right\n\t\t\t float fScale) {\n\n')

    f.write('\tfloat ul = ul_in / 256.0;\n')
    f.write('\tfloat um = um_in / 256.0;\n')
    f.write('\tfloat ur = ur_in / 256.0;\n')
    f.write('\tfloat ml = ml_in / 256.0;\n')
    f.write('\tfloat mm = mm_in / 256.0;\n')
    f.write('\tfloat mr = mr_in / 256.0;\n')
    f.write('\tfloat ll = ll_in / 256.0;\n')
    f.write('\tfloat lm = lm_in / 256.0;\n')
    f.write('\tfloat lr = lr_in / 256.0;\n')
    f.write('\n')

    f.write('\tfloat sum = 0.0;\n')

    inputs = ['ul', 'um', 'ur', 'ml', 'mm', 'mr', 'll', 'lm', 'lr', 'BIAS']
    for i in range(1, numLayers):
        for neuron in labels[sum(layers[:i]):sum(layers[:i+1])]:
            f.write('\tsum = 0\n')
            j = 0
            for inNeuron in labels[sum(layers[:i-1]):sum(layers[:i])]:
                if i == 1:
                    f.write('\t+ ' + inputs[j] + '*w__' + inNeuron + '__' + neuron + '\n')
                    j += 1
                else :
                    f.write('\t+ n__' + inNeuron + '*w__' + inNeuron + '__' + neuron + '\n')
            f.write(';\n')
            f.write('\tfloat n__' + neuron + ' = af__' + neuron + '(sum, as__' + neuron + ');\n')

    f.write('\treturn (unsigned char) (n__2_0 * 256.0);\n}\n#endif\n')

    f.close()

def main(): 
    f = '../../fann.configs/sobel_FANN.nn'
    config = open(f, 'r')

    lines = config.readlines()

    numLayers = int(lines[1].split('=')[1])

    layers = []

    l = lines[-4].split('=')[1].split(' ')
    for i in range(numLayers):
        layers.append(int(l[i]))

    neuronsLine = lines[-2]
    triplets = neuronsLine.split('=(')[1].split(') (')
    triplets[-1] = triplets[-1].split(')')[0]

    numInputs = []
    af = []
    steep = []
    labels = []
    for i in range(sum(layers)):
        for j in range(len(layers)):
            if i < sum(layers[:j+1]):
                labels.append(str(j) + '_' + str(i - sum(layers[:j])))
                break
        numInputs.append(int(triplets[i].split(',')[0]))
        af.append(int(triplets[i].split(',')[1]))
        steep.append(float(triplets[i].split(',')[2]))

    weights = {}
    connections = []
    for i in range(1, len(layers)):
        for j in range(sum(layers[:i]), sum(layers[:i])+layers[i]):
            for k in range(sum(layers[:i-1]), sum(layers[:i-1])+layers[i-1]):
                weights[labels[k] + '__' + labels[j]] = 0.0
                connections.append(labels[k] + '__' + labels[j])

    non0connections = []
    for c in connections:
        label = c.split('__')[1]
        layer = int(label.split('_')[0])
        row = int(label.split('_')[1])
        i = sum(layers[:layer]) + row
        if not numInputs[i] == 0:
            non0connections.append(c)

    connectionsLine = lines[-1]
    pairs = connectionsLine.split('=(')[1].split(') (')
    pairs[-1] = pairs[-1].split(')')[0]

    i = 0
    for c in non0connections:
        weights[c] = float(pairs[i].split(',')[1])
        i += 1
    
    printHeader(af, steep, weights, labels) 
    printKernel(numLayers, labels, layers)

    config.close()

if __name__ == '__main__':
    main()

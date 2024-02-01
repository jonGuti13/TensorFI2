import numpy as np
import glob
import cv2

carpetaPrediccionFI = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/PredTF2/Float/WholeExp104/todo_224_416_TC_PN_5_3_32_fold4_init1_explicit_different_norm_wd_wd_5_ratio1_5_ratio2_fault_injection/"
carpetaPrediccionNFI = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/PredTF2/Float/WholeExp104/todo_224_416_TC_PN_5_3_32_fold4_init1_explicit_different_norm_wd_wd_5_ratio1_5_ratio2/"
carpetaGT = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/Labels_224_416/Exp104/"

# Sacar cuántas veces se ha hecho flip en cada uno de los píxeles en cada una de las capas

files = glob.glob(carpetaPrediccionFI + "*.png")
bitFlipPositions = np.zeros((100, 32), dtype=int) #100 capas, 9 bitFlipPositions (ojo el MSB esta a la derecha)
for file in files:
    filename = file.split("/")[-1].split(".")[0] #nfXXXX_XXX_YYYY_YYY_YY
    layerIdx = int(filename.split("_")[3])
    bitIdx = int(filename.split("_")[4])
    bitFlipPositions[layerIdx][bitIdx] += 1

#bitFlipPositions = bitFlipPositions * 10 #Si quiero tener en cuenta todas las imágenes


# files = glob.glob(carpetaPrediccionFI + "*.png")
# files.sort()
# bitFlipError = np.zeros((100, 32), dtype=int) #100 capas, 9 bitFlipPositions (ojo el MSB esta a la derecha)
# bitFlipErrorReduced = np.zeros((100, 32), dtype=int) #100 capas, 9 bitFlipPositions (ojo el MSB esta a la derecha)
# bitFlipErrorReduced2 = np.zeros((100, 32), dtype=int) #100 capas, 9 bitFlipPositions (ojo el MSB esta a la derecha)
# for file in files:
#     filename = file.split("/")[-1].split(".")[0] #nfXXXX_XXX_YY_YYY_YY
#     prediccionFI = np.array(cv2.imread(file, 1)[:, : , 0])
#     prediccionNFI = np.array(cv2.imread(carpetaPrediccionNFI + "/" + filename.split("_")[0] + "_" + filename.split("_")[1] + ".png", 1)[:, : , 0])
#     labelGT = np.array(cv2.imread(carpetaGT + "/" + filename.split("_")[0] + "_" + filename.split("_")[1] + ".png", 1)[:, : , 0])

#     layerIdx = int(filename.split("_")[3])
#     bitIdx = int(filename.split("_")[4])

#     bitFlipError[layerIdx][bitIdx] += np.sum((prediccionNFI != prediccionFI))
#     bitFlipErrorReduced[layerIdx][bitIdx] += np.sum(prediccionNFI[8:, 7:] != prediccionFI[8:, 7:])
#     bitFlipErrorReduced2[layerIdx][bitIdx] += np.sum((prediccionNFI != prediccionFI) & (labelGT != 0))

np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipPositionsPruned.npy", bitFlipPositions)
#np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipErrorPruned.npy", bitFlipError)
#np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipErrorReducedPruned.npy", bitFlipErrorReduced)
#np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipErrorReduced2Pruned.npy", bitFlipErrorReduced2)

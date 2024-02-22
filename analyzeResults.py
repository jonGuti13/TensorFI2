import numpy as np
import glob
import cv2

#PRUNED
#carpetaPrediccionFI = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/PredTF2/Quant/WholeExp104/todo_192_384_TC_PN_5_3_32_fold4_init1_different_norm_wd_wd_tf2_train_5_ratio1_5_ratio2_fault_injection_tflite/"
#carpetaPrediccionNFI = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/PredTF2/Quant/WholeExp104/todo_192_384_TC_PN_5_3_32_fold4_init1_different_norm_wd_wd_tf2_train_5_ratio1_5_ratio2_tflite"
#NOT PRUNED
carpetaPrediccionFI = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/PredTF2/Quant/WholeExp104/todo_192_384_TC_PN_5_3_32_fold4_init1_different_wd_wd_fault_injection_tflite_sign/"
carpetaPrediccionNFI = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/PredTF2/Quant/WholeExp104/todo_192_384_TC_PN_5_3_32_fold4_init1_different_wd_wd_tflite"
carpetaGT = "/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/Labels_192_384/Exp104/"

# Sacar cuántas veces se ha hecho flip en cada uno de los píxeles en cada una de las capas

files = glob.glob(carpetaPrediccionFI + "*.png")
bitFlipPositions = np.zeros((100, 32), dtype=int) #100 capas (not quantized) o 56 capas (quantized o folded), 9 bitFlipPositions (ojo el MSB esta a la derecha) #56 capas para el modelo cuantizado
for file in files:
    filename = file.split("/")[-1].split(".")[0] #nfXXXX_XXX_YYYY_YYY_YY o #nfXXXX_XXX_patch_XXXX_YYYY_YYY_YY
    layerIdx = int(filename.split("_")[3])       #                 3     o                              5
    bitIdx = int(filename.split("_")[4])         #                 4     o                              6
    bitFlipPositions[layerIdx][bitIdx] += 1


files = glob.glob(carpetaPrediccionFI + "*.png")
files.sort()
bitFlipError = np.zeros((100, 32), dtype=int) #100 capas (not quantized) o 56 capas (quantized  o folded), 9 bitFlipPositions (ojo el MSB esta a la derecha)
bitFlipErrorReduced = np.zeros((100, 32), dtype=int) #100 capas (not quantized) o 56 capas (quantized  o folded), 9 bitFlipPositions (ojo el MSB esta a la derecha)
bitFlipErrorReduced2 = np.zeros((100, 32), dtype=int) #100 capas (not quantized) o 56 capas (quantized  o folded), 9 bitFlipPositions (ojo el MSB esta a la derecha)
for file in files:
   filename = file.split("/")[-1].split(".")[0] #nfXXXX_XXX_YY_YYY_YY
   prediccionFI = np.array(cv2.imread(file, 1)[:, : , 0])
   prediccionNFI = np.array(cv2.imread(carpetaPrediccionNFI + "/" + filename.split("_")[0] + "_" + filename.split("_")[1] + ".png", 1)[:, : , 0])
   labelGT = np.array(cv2.imread(carpetaGT + "/" + filename.split("_")[0] + "_" + filename.split("_")[1] + ".png", 1)[:, : , 0])
   prediccionNFI = np.array(cv2.imread(carpetaPrediccionNFI + "/" + filename.split("_")[0] + "_" + filename.split("_")[1] + "_"  + filename.split("_")[2] + "_" + filename.split("_")[3] + ".png", 1)[:, : , 0])
   labelGT = np.array(cv2.imread(carpetaGT + "/" + filename.split("_")[0] + "_" + filename.split("_")[1] + "_" + filename.split("_")[2] + "_" + filename.split("_")[3] + ".png", 1)[:, : , 0])

   layerIdx = int(filename.split("_")[3])
   bitIdx = int(filename.split("_")[4])

   bitFlipError[layerIdx][bitIdx] += np.sum((prediccionNFI != prediccionFI))
   bitFlipErrorReduced[layerIdx][bitIdx] += np.sum(prediccionNFI[8:, 7:] != prediccionFI[8:, 7:])
   bitFlipErrorReduced2[layerIdx][bitIdx] += np.sum((prediccionNFI != prediccionFI) & (labelGT != 0))

np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipPositionsNotPruned.npy", bitFlipPositions)
#np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipErrorNotPruned.npy", bitFlipError)
np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipErrorNotReducedPruned.npy", bitFlipErrorReduced)
np.save("/workspace/Vitis-AI/tutorials/HSI-DRIVE/files/workspace/baseDeDatos/HSI-Drive_2.0_Jon/Fold4/bitFlipErrorNotReduced2Pruned.npy", bitFlipErrorReduced2)

 Q1: Feature Vector Length  
Setting A (`8x8`) → more cells → longer vector, finer details.  
Setting B (`16x16`) → fewer cells → shorter vector, coarser info.  

---
 Q2: Linear SVM vs. k-NN  
SVM: 46.67% accuracy.  
k-NN: 40.00% accuracy.  
SVM wins because it learns better boundaries; k-NN struggles with noise and small data.  

---
 Q3: Most Confused Classes  
Dogs ↔ Pandas misclassified.  
Similar shapes/textures cause overlap in HOG features.  
More diverse training images could reduce confusion.  

---
 Q4: Real-World Application  
HOG + classical models suit factory quality control.  
Efficient, low-cost, real-time inspection without heavy hardware.  

---

Would you like me to compress these even further into **one-line answers per question**?

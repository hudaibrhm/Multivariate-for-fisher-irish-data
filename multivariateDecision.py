# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 07:31:01 2017

@author: Huda Ibrhm
"""
import csv
import numpy
from scipy.stats import multivariate_normal

f1 = open('dataTraining.csv')
csvf_1 = csv.reader(f1)

n = []
#print(n)
for row in csvf_1:
    #print(row[2])
    n.append(row)

ndata = len(n)
print(ndata)


A1 = []
A2 = []
for b1 in n:
    A1.append(b1[1])
    A2.append(b1[2])

#print(A1)

A1setosa = []
A1versi = []
A2setosa = []
A2versi = []
for x in range(1,ndata):
    if x<=40:
        A1setosa.append(float(A1[x]))
        A2setosa.append(float(A2[x]))                
    else:
        A1versi.append(float(A1[x]))
        A2versi.append(float(A2[x]))
        
        
#print(A1setosa)
f2 = open('dataTesting.csv')
csvf_2 = csv.reader(f2)

ntest = []
for row in csvf_2:
    ntest.append(row)

ndatatest = len(ntest)
#print(ndatatest)

A1test = []
A2test = []
for b2 in ntest:
    A1test.append(b2[1])
    A2test.append(b2[2])
#print(A2test)

A1testsetosa = []
A1testversi = []
A2testsetosa = []
A2testversi = []
for x in range(0,ndatatest):
    if x<10:
        A1testsetosa.append(float(A1test[x]))
        A2testsetosa.append(float(A2test[x]))                
    else:
        A1testversi.append(float(A1test[x]))
        A2testversi.append(float(A2test[x]))
        
ndatatestsetosa = len(A1testsetosa);
ndatatestversi = len(A1testversi);

"""
print(ndatatestsetosa)
print(ndatatestversi)
        
print(A1testsetosa)
print(A1testversi)
"""

f1.close()
f2.close()

"""
Code Untuk Algoritma Bayes Decision
"""

#menghitung  nilai rata2 setiap data training
mx0A1 = numpy.mean(A1setosa)
mx0A2 = numpy.mean(A2setosa)

mx1A1 = numpy.mean(A1versi)
mx1A2 = numpy.mean(A2versi)

#menghitung nilai variansi dari data training
varx0A1 = numpy.var(A1setosa)
varx0A2 = numpy.var(A2setosa)

varx1A1 = numpy.var(A1versi)
varx1A2 = numpy.var(A2versi)


def cov(a, b):

    if len(a) != len(b):
        return

    a_mean = numpy.mean(a)
    b_mean = numpy.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)
    
s011 = cov(A1setosa,A1setosa) 
s012 = cov(A1setosa,A2setosa)
s021 = s012
s022 = cov(A2setosa,A2setosa)

s111 = cov(A1versi,A1versi) 
s112 = cov(A1versi,A2versi)
s121 = s012
s122 = cov(A2versi,A2versi)

#print(s11)
X1 = [A1setosa,A2setosa]
X2 = [A1versi,A2versi]

M1 = [mx0A1,mx0A2]
M2 = [mx1A1,mx1A2]


S1setosa = [[s011,s012],[s021,s022]]
S1versi = [[s111,s112],[s121,s122]]

"""
temp1 = numpy.linalg.det(S1setosa)
temp2 = numpy.linalg.det(S1setosa)
d = 2
setosasatuPer = 1 /(((2*3.14)**(d/2))*numpy.sqrt(temp1))
versisatuPer = 1 /(((2*3.14)**(d/2))*numpy.sqrt(temp2))
"""
phasil0 = []
phasil1 = []
#temp1 = multivariate_normal(mean,cov)
#pa = temp1.pdf(thedata[i])
temp1 = multivariate_normal(M1,S1setosa)
temp2= multivariate_normal(M2,S1versi)


for x in range (0,ndatatestsetosa): 
    phasil0.append(temp1.pdf([A1testsetosa[x],A2testsetosa[x]]))
    phasil1.append(temp2.pdf([A1testversi[x],A2testversi[x]]))
    
    #phasil1[x] = versisatuPer*numpy.exp((-1/2)*numpy.transpose(X2[x]-M2)*numpy.linalg.inv(S1versi)*(X2[x]-M2))

phasil0v2 = []
phasil1v2 = []    
for x in range(0,ndatatestsetosa):
    phasil0v2.append(temp2.pdf([A1testsetosa[x],A2testsetosa[x]]))
    phasil1v2.append(temp1.pdf([A1testversi[x],A2testversi[x]]))
    
    
#mencetak nilai2 training
print("Nilai data Training :")
print("A1 Bunga Setosa :",A1setosa)
print()
print("A2 Bunga Setosa :",A2setosa)
print()
print("A1 Bunga Versicolor :",A1versi)
print()
print("A2 Bunga Versicolor :",A2versi)
print()

#mencetak nilai rata2 data training
print("Rata2 data training :")
print(mx0A1)
print(mx0A2)
print(mx1A1)
print(mx1A2)
print()

#mencetak nilai variansi data training
print("Varians data training :")
print(varx0A1)
print(varx0A2)
print(varx1A1)
print(varx1A2)
print()

#mencetak nilai2 data testing
print("Nilai data Testing :")
print("A1 Bunga Setosa :",A1testsetosa)
print()
print("A2 Bunga Setosa :",A2testsetosa)
print()
print("A1 Bunga Versicolor :",A1testversi)
print()
print("A2 Bunga Versicolor :",A2testversi)

#mengoutputkan hasil p dari perhitungan        
print("Hasil P setosa = ",phasil0);
print()
print("Hasil P setosa v2 =",phasil0v2); 
print()   
print("Hasil P versicolor = ",phasil1);
print()
print("Hasil P versicolor v2 = ",phasil1v2);
print() 


countTrue = 0
print("hasil Perbandingan Setosa && SetosaV2 =")
for x in range(0,ndatatestsetosa):
    a = phasil0[x]
    b = phasil0v2[x]
    if a > b :
        print("TRUE")
        countTrue += 1
    else:
        print("FALSE")
        
print()
print("hasil Perbandingan Versicolor && VersicolorV2 =")
for x in range(0,ndatatestsetosa):
    if phasil1[x] > phasil1v2[x] :
        print("TRUE")
        countTrue += 1
    else:
        print("FALSE") 
        
print()

akurasi = (countTrue/ndatatest)*100
#menghitung Akurasi
print("Jumlah Data Testing yang sesuai dengan kelasnya : ",countTrue)
print("Akurasi yang didapatkan : ",akurasi,"%")            
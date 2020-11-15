#!/usr/bin/python
from __future__ import division
from numba import cuda, float32
import numpy as np
import math 

TPB = 5

class glogic:
  def __init__(self,array,matrix):
    self.array = array
    self.matrix = matrix

  def set_matrix_value(self,i,j,value):
      self.matrix[i][j] = value
  
  def gnot(self,value1,value2):
      out = abs(value1-value2)
      return out

  def gnot_single_value(self,value):
      max_value = max(self.array)
      out = abs(max_value-value)
      return out

  def gand(self,value1,value2):
      out = math.floor((value1 + value2)/2) 
      return out
  
  def gnand(self,value1,value2):
      out = max(self.array) - math.floor((value1 + value2)/2) 
      return out

  def gor(self,value1,value2):
      out = math.ceil((value1 + value2)/2)
      return out    

  def gnor(self,value1,value2):
      out = max(self.array) - math.ceil((value1 + value2)/2)
      return out    

  def gexor(self,value1,value2):
      if (value1 != value2):
          if value1 < value2:
              return value1
          else:
              return value2
      else:
          temp = max(self.array) - math.floor((value1+value2)/2)
          return temp     
    
  def gexand(self,value1,value2):
      if (value1 != value2):
          if value1 > value2:
              return value1
          else:
              return value2
      else:
          temp = math.floor((value1+value2)/2)
          return temp  



  def init_matrix(self):
      num_of_rows = max(self.array)+1
      num_of_cols = max(self.array)+1
      self.matrix = np.full((num_of_rows, num_of_cols), np.zeros) 

  def print_gnot_matrix(self):
      for x in self.array:
          for y in self.array:
              prova = self.gnot(x,y)
              self.set_matrix_value(x,y,prova)
        
      print("Gemma logic NOT Matrix")
      print(self.matrix)

  def print_gand_matrix(self):
      for x in self.array:
          for y in self.array:
              prova = self.gand(x,y)
              self.set_matrix_value(x,y,prova)
        
      print("Gemma logic AND Matrix")
      print(self.matrix)

  def print_gor_matrix(self):
      for x in self.array:
          for y in self.array:
              prova = self.gor(x,y)
              self.set_matrix_value(x,y,prova)
        
      print("Gemma logic OR Matrix")
      print(self.matrix)     
      
  def print_gnand_matrix(self):
      for x in self.array:
          for y in self.array:
              prova = self.gnand(x,y)
              self.set_matrix_value(x,y,prova)
        
      print("Gemma logic NAND Matrix")
      print(self.matrix)    
  
  def print_gnor_matrix(self):
      for x in self.array:
          for y in self.array:
              prova = self.gnor(x,y)
              self.set_matrix_value(x,y,prova)
        
      print("Gemma logic NOR Matrix")
      print(self.matrix)
  
  def print_gexor_matrix(self):
      for x in self.array:
          for y in self.array:
              prova = self.gexor(x,y)
              self.set_matrix_value(x,y,prova)
        
      print("Gemma logic EXOR Matrix")
      print(self.matrix)

  def print_gexand_matrix(self):
      for x in self.array:
          for y in self.array:
              prova = self.gexand(x,y)
              self.set_matrix_value(x,y,prova)
        
      print("Gemma logic EXAND Matrix")
      print(self.matrix)   

  def vector_gnot(self,vector):
      temp = vector
      if np.size(temp)==1:
          out = self.gnot_single_value(temp)
          return out
      else:
          #print(temp) 
          iter = len(temp)-1
          range_data = len(temp)-1
          #print(iter)
          while iter>=1:
              for x in range(0,range_data,1):
                  temp[x] = self.gnot(temp[x],temp[x+1])
                  
              print(temp)
              range_data = range_data -1
              iter = iter - 1
              #print(iter)
              
          out = temp[0]
      return out

  def vector_gand(self,vector):
      #print(vector) 
      temp = vector
      iter = len(temp)-1
      range_data = len(temp)-1
      #print(iter)
      while iter>=1:
            for x in range(0,range_data,1):
                temp[x] = self.gand(temp[x],temp[x+1])
                
            print(temp)  
            range_data = range_data -1
            iter = iter - 1
            #print(iter)
              
      out = temp[0]
      return out

  def vector_gor(self,vector):
      ##print(vector) 
      iter = len(vector)-1
      range_data = len(vector)-1
      #print(iter)
      while iter>=1:
            for x in range(0,range_data,1):
                vector[x] = self.gor(vector[x],vector[x+1])
                
            print(vector)
              
            range_data = range_data -1
            iter = iter - 1
            #print(iter)
              
      out = vector[0]
      return out

  def vector_gnand(self,vector):
      ##print(vector) 
      iter = len(vector)-1
      range_data = len(vector)-1
      #print(iter)
      while iter>=1:
            for x in range(0,range_data,1):
                vector[x] = self.gnand(vector[x],vector[x+1])
                ##print(vector)
              
            range_data = range_data -1
            iter = iter - 1
            #print(iter)
              
      out = vector[0]
      return out       

  def vector_gnor(self,vector):
      ##print(vector) 
      iter = len(vector)-1
      range_data = len(vector)-1
      #print(iter)
      while iter>=1:
            for x in range(0,range_data,1):
                vector[x] = self.gnor(vector[x],vector[x+1])
                ##print(vector)
              
            range_data = range_data -1
            iter = iter - 1
            #print(iter)
              
      out = vector[0]
      return out

  def vector_gexor(self,vector):
      ##print(vector) 
      iter = len(vector)-1
      range_data = len(vector)-1
      #print(iter)
      while iter>=1:
            for x in range(0,range_data,1):
                vector[x] = self.gexor(vector[x],vector[x+1])
                ##print(vector)
              
            range_data = range_data -1
            iter = iter - 1
            #print(iter)
              
      out = vector[0]
      return out   

  def vector_gexand(self,vector):
      ##print(vector) 
      iter = len(vector)-1
      range_data = len(vector)-1
      #print(iter)
      while iter>=1:
            for x in range(0,range_data,1):
                vector[x] = self.gexand(vector[x],vector[x+1])
                ##print(vector)
              
            range_data = range_data -1
            iter = iter - 1
            #print(iter)
              
      out = vector[0]
      return out     






  











  



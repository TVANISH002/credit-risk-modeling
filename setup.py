from setuptools import find_packages, setup
from typing import List

def get_requiremnts()-> List[str]:
     """
     This function will return list of requirements"""
     requirement_lst:List[str] = [] 
     
     try:
          with open("requirements.txt","r") as file:
               #Read lines from file
               lines = file.readlines()
               #process each line
               for line in lines:
                    requirement = line.strip()
                    ##ignore empty lines and -e. 
                    if requirement and requirement!= '-e .':
                         requirement_lst.append(requirement)
     except FileNotFoundError:
          print("Your requirement.txt file not found")         
          
     return requirement_lst   
#print(get_requiremnts())
 
 #setting up meta data
 
setup(
      name = "networksecurity",
      version="0.0.1",
      author = "Anish Tirumala Venkata",
      author_email = "anishve9@gmail.com",
      packages = find_packages(),
      install_requires = get_requiremnts() #this make sures when packages are gettign build it install required libraries
      
      
 )
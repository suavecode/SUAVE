
def whav(x1,y1,x2,y2):

#<<<<<<< .mine
    if x1==x2:
        whv=1/(y1-y2)
    else:  
        whv=1/(y1-y2)*(1+ (numpy.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))
    
    return whv
#=======
#def whav(x1,y1,x2,y2):
#>>>>>>> .r226

    #if x1==x2:
        #whv=1/(y1-y2)
    #else:  
        #whv=1/(y1-y2)*(1+ (numpy.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))
    
    #return whv

import numpy as np
from sklearn.svm import LinearSVC

def my_fit(Z_train):
    F=1040
    new_train = np.ones((np.shape(Z_train)[0],F+1))
    for a in range(65):
        for i in range(16):
            if(a!=64):
                new_train[:,a*16 + i] = np.multiply(new_train[:,a*16 + i],Z_train[:,a])
            i1 = i
            a1 = np.ones(np.shape(Z_train)[0],)
            a2 = np.ones(np.shape(Z_train)[0],)
            for t in range(4):
                r = i1%2
                i1=i1//2
                if r:
                    a1 = np.multiply(a1,Z_train[:,64+t])
                    a2 = np.multiply(a2,Z_train[:,68+t])
                else:
                    a1 = np.multiply(a1,1-Z_train[:,64+t])
                    a2 = np.multiply(a2,1-Z_train[:,68+t]) 
            new_train[:,a*16 + i] = np.multiply(new_train[:,a*16 + i],a1-a2)

    new_train[:,-1] = Z_train[:,-1]
    
    clf = LinearSVC(loss="squared_hinge",max_iter=5000,tol=0.1,C=100)
    clf.fit(new_train[ :, :F ], new_train[ :, -1 ])
    return clf


def my_predict( X_tst, model):
    F=1040
    new_test = np.ones((np.shape(X_tst)[0],F))
    for a in range(65):
        for i in range(16):
            if(a!=64):
                new_test[:,a*16 + i] = np.multiply(new_test[:,a*16 + i],X_tst[:,a])
            i1 = i
            a1 = np.ones(np.shape(X_tst)[0],)
            a2 = np.ones(np.shape(X_tst)[0],)
            for t in range(4):
                r = i1%2
                i1=i1//2
                if r:
                    a1 = np.multiply(a1,X_tst[:,64+t])
                    a2 = np.multiply(a2,X_tst[:,68+t])
                else:
                    a1 = np.multiply(a1,1-X_tst[:,64+t])
                    a2 = np.multiply(a2,1-X_tst[:,68+t])                 
            new_test[:,a*16 + i] = np.multiply(new_test[:,a*16 + i],a1-a2)
        
    return model.predict(new_test)
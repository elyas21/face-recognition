import redis
import cv2
import time
face_list = set()

def save_img(img, bbox, identities=None, offset=(0,0)):

    for i,box in enumerate(bbox):
        
        id = int(identities[i]) if identities is not None else 0    

        if(id not in face_list):
            face_list.add(id)
            cv2.imwrite('output/img/'+str(id)+'.jpg', img=img)
            print('face reeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')

    return img

def saveToRedis(id,r,name):
    getPath = str(str(time.time()))[:9]+str(id)
    r.set(getPath,name)


    
# if __name__=='__main__':
#     saveToDB()
    
import redis
import cv2
face_list = set()

def save_img(img, bbox, identities=None, offset=(0,0)):

    for i,box in enumerate(bbox):
        
        id = int(identities[i]) if identities is not None else 0    

        if(id not in face_list):
            face_list.add(id)
            cv2.imwrite('output/img/'+str(id)+'.jpg', img=img)
            print('face reeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')

    return img

# def saveToDB():
#     r = redis.Redis(
#     host='redis-16109.c328.europe-west3-1.gce.redns.redis-cloud.com',
#     port=16109,
#     password='----')
    
#     r.set('name','elyas')
#     print(r.get('name'))
#     r.set('name','elyasss')
#     print(r.get('name'))

    
# if __name__=='__main__':
#     saveToDB()
    
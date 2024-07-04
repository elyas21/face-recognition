import redis

def saveToDB():
    r = redis.Redis(
    host='redis-16109.c328.europe-west3-1.gce.redns.redis-cloud.com',
    port=16109,
    password='bawEnIPG6hRrMNTs6lyQ0NF6tQq05nso')
    
    r.set('name','elyas')
    print(r.get('name'))
    r.set('name','elyasss')
    print(r.get('name'))

    
if __name__=='__main__':
    saveToDB()
    
import sys , pygame, pygame.gfxdraw

import math

def sig(n):
    if n%2==0:return 1
    return-1

class Vec3:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z

    def __str__(self):
        return (f"({self.x}, {self.y}, {self.z})")
    
    def __add__(self,other):
        return Vec3(self.x+other.x,self.y+other.y,self.z+other.z)

    def __sub__(self,other):
        return Vec3(self.x-other.x,self.y-other.y,self.z-other.z)

    def __rmul__(self,f):
        if isinstance(f, Matrix):
            return f * Matrix([[self.x],[self.y],[self.z],[1]])

        return Vec3(f*self.x,f*self.y,f*self.z)

    def __mul__(self,other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __xor__(self,other):
        return Vec3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

    @property
    def norm(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    @property
    def normalized(self):
        return (1/self.norm)*self

    def proj(self, f):
        return (self.x*f/self.z, self.y*f/self.z)

class Matrix:
    def __init__(self,values):
        self.values=values
    
    @property
    def width(self):
        return len(self.values[0])
    
    @property
    def height(self):
        return len(self.values)

    def __getitem__(self,t):
        c,l = t
        return self.values[l][c]
    
    def __str__(self):
        res = ""
        for l in self.values:
            for c in l:
                res += str(c) + " "
            res += "\n"
        return res

    @staticmethod
    def identity(n):
        return Matrix([[ (1 if l==c else 0) for c in range(n)  ] for l in range(n)])

    def __mul__(self,other):
        if isinstance(other, Vec3):
            m = self * Matrix([[other.x],[other.y],[other.z],[1]])
            return Vec3(m[0,0],m[0,1],m[0,2])
        if self.width!=other.height:
            raise Exception("les matrices ne sont pas compatibles")
        return Matrix(
            [ [sum([(self[i,l] * other[c,i]) for i in range(self.width)]) for c in range(other.width)] for l in range(self.height)])

    def __rmul__(self, x):
        return Matrix(
            [ [ x * self[c,l] for c in range(self.width) ] for l in range(self.height) ] 
        )

    @staticmethod
    def rotx(t):
        sint=math.sin(t)
        cost=math.cos(t)
        return Matrix([
            [1,0,0,0],
            [0,cost,-sint,0],
            [0,sint,cost,0],
            [0,0,0,1]

        ])
    @staticmethod
    def roty(t):
        sint=math.sin(t)
        cost=math.cos(t)
        return Matrix([
            [cost,0,sint,0],
            [0,1,0,0],
            [-sint,0,cost,0],
            [0,0,0,1]

        ])
    @staticmethod
    def rotz(t):
        sint=math.sin(t)
        cost=math.cos(t)
        return Matrix([
            [cost,-sint,0,0],
            [sint,cost,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])
    @staticmethod
    def rot(tx,ty,tz) :
        return Matrix.rotx(tx)*Matrix.roty(ty)*Matrix.rotz(tz)

    @staticmethod
    def transl(x,y,z):
        return Matrix([
            [1,0,0,x],
            [0,1,0,y],
            [0,0,1,z],
            [0,0,0,1]
        ])
    @staticmethod
    def scale(x=1,y=1,z=1):
        return Matrix([
            [x,0,0,0],
            [0,y,0,0],
            [0,0,z,0],
            [0,0,0,1]
        ])

    @staticmethod
    def cam(camera,target,up=Vec3(0,1,0)):
        uz = (camera-target).normalized
        ux = (up ^ uz).normalized
        uy = uz ^ ux
        return Matrix(
            [[ux.x,uy.x,uz.x, camera.x],
             [ux.y,uy.y,uz.y, camera.y],
             [ux.z,uy.z,uz.z, camera.z],
             [   0,   0,  0,         1]]
        ).inv

    def sub(self,c,l):
        w = self.width
        h = self.height
        return Matrix(
            [ [self[ci,li] for ci in range(w) if ci!=c ] for li in range(h) if li!=l ]
        )

    @property
    def det(self):
        if self.width!=self.height:
            raise Exception("Cette matrice n'a pas de determinant")
        if self.width==1:return self[0,0]
        d=0
        s=1
        for i in range(self.width):
            d+=s*self[i,0]*self.sub(i,0).det
            s=-s
        return d

    
    @property
    def transp(self):
        return Matrix([
            [ self[l,c] for c in range(self.height)] for l in range(self.width)
        ])
        

    @property
    def co(self):
        return Matrix([
            [ (self.sub(c,l).det)*sig(c+l) for c in range(self.width)] for l in range(self.height)
        ])

    @property
    def inv(self):
        return (1/self.det)*self.co.transp

pygame.init()


size = width, height = 1150,900
speed = [2,2]
black = 0,0,0
f=700

screen = pygame.display.set_mode(size)
cube = [ Vec3(i-0.5,j-0.5,k-0.5) for i in range(2) for j in range (2) for k in range(2)   ]
plan = [ Vec3(-1,0,-1), Vec3(-1,0,1), Vec3(1,0,-1), Vec3(1,0,1) ]
def line(ps, p1,p2):
    x1,y1 = ps[p1]
    x2,y2 = ps[p2]

    pygame.gfxdraw.line(screen, int(x1+width/2), int(y1+height/2), int(x2+width/2) , int(y2+height/2), (255,255,255) )


while 1:
    for event in pygame.event.get():
       if event.type == pygame.QUIT: sys.exit()
    
    screen.fill(black)
    t = pygame.time.get_ticks() / 2000
    m = Matrix.transl(0,25,0) * Matrix.rot(t*2,-t,-t*1.5) * Matrix.scale(20,20,20)
    s = Matrix.scale(20,20,20)
    wcube = [ (m * p) for p in cube]
    wplan = [ (s * p) for p in plan]
    w = wcube + wplan
    cam = Matrix.cam(Vec3(80*math.sin(t),30+math.sin(t*2)*20,80*math.cos(t)),Vec3(0,0,0))
    r = [ (cam * p).proj(f) for p in w ]
    for px,py in r:
        pygame.gfxdraw.circle(screen,int(px+width/2),int(py+height/2),3, (255,255,255))

    for i in range(4):
        line(r,i,i+4)
        line(r,i*2,i*2+1)
    line(r,0,2)
    line(r,1,3)
    line(r,4,6)
    line(r,5,7)

    line(r,8,9)
    line(r,8,10)
    line(r,9,11)
    line(r,10,11)
     

    pygame.display.flip()



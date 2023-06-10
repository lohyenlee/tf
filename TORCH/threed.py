import numpy as np
from pythreejs import *
import PIL.Image                     # for importing PNG files

#======================================================================
#
#  CRUCIAL HELPER FUNCTIONS
#
#=====================================================================  
def normalize (v): return v/np.linalg.norm(v)

def findNormal (u):
  if u[1]!=0: return normalize(np.cross ([1,0,0], u))
  if u[2]!=0: return normalize(np.cross ([0,1,0], u))
  if u[0]!=0: return normalize(np.cross ([0,0,1], u)) 
  raise Exception ("findNormal([0,0,0]) is illegal!")

def quatUToV (u,v): # find a quaternion that describes a rotation that rotates u to v
  u = normalize(u)
  v = normalize(v)
  udotv = u@v
  if udotv==1:  
    return (0,0,0,1)     # identity
  elif udotv==-1: 
    t = np.pi            # 180-degree rotation
    a = findNormal (u)   # any axis perpendicular to u
  else:  
    t = np.arccos(udotv)         # rotation angle
    a = normalize(np.cross(u,v)) # rotation axis
  s = np.sin(t/2); c = np.cos(t/2)
  return (s*a[0], s*a[1], s*a[2], c)  # convert axis-angle (a,t) to quaternion


#======================================================================
#
#  GEOMETRIC PRIMITIVES AND RENDERING CODE
#  render, sphere, cylinder, billboard
#
#=====================================================================  
def lookAt (forward,up): # find a quaternion that describes a rotation that rotates [0,0,1] to forward while preserving something about the up direction
  c = -normalize(forward)
  a = normalize(np.cross(up, c)) # not sure about all this
  b = normalize(np.cross(c, a))
  return np.array ([a,b,c]).T
def copysign (x, s):
  return x * np.sign(x*s)
def quatFromRotMat (m):
  w = sqrt( max( 0, 1 + m[0,0] + m[1,1] + m[2,2] ) ) / 2
  x = sqrt( max( 0, 1 + m[0,0] - m[1,1] - m[2,2] ) ) / 2
  y = sqrt( max( 0, 1 - m[0,0] + m[1,1] - m[2,2] ) ) / 2
  z = sqrt( max( 0, 1 - m[0,0] - m[1,1] + m[2,2] ) ) / 2
  x = copysign( x, m[2,1] - m[1,2] )
  y = copysign( y, m[0,2] - m[2,0] )
  z = copysign( z, m[1,0] - m[0,1] )
  return (x,y,z,w)
  
  
def render (objects, cameraPosition=[0,-20,20], fov=55, viewCenter=[0,0,0], imageSize=[640,480]):
  width,height = imageSize
  lightD = DirectionalLight(color='#FFFFFF', position=[3,5,1], intensity=1.0) # directional light attached to camera
  lightA = AmbientLight(color='#333333')
  camera = PerspectiveCamera(position=cameraPosition, fov=fov, aspect=width/height, children=[lightD,lightA])
  camera.up = [0,0,1]
  scene = Scene(children=[*objects, camera])
  control = OrbitControls(controlling=camera,target=tuple(viewCenter))
  viewDirection = np.array(viewCenter) - cameraPosition 
  m = lookAt (viewDirection, camera.up)
  camera.quaternion = quatFromRotMat (m)   # this code is better
  #camera.lookAt (tuple(viewCenter))   # this three.js commands cause minor problems with pythreejs
  renderer = Renderer(scene, camera, controls=[control], width=width, height=height, antialias=True)
  return renderer


geomSphere = SphereBufferGeometry(1.0, 24, 24)
def sphere (position=[0,0,0], radius=1.0, color='#999999'): #, **kwargs):
  position = np.array(position)
  material = MeshPhongMaterial (color=color)
  mesh = Mesh(geomSphere, material)
  mesh.position = tuple(position)
  mesh.scale = [radius,radius,radius]
  return mesh
def cylinder (rA, rB, radius, color='#999999', radialSegments=6):  #**kwargs):
  radius = radius * 1.0
  rA = np.array(rA)
  rB = np.array(rB)
  geometry = CylinderBufferGeometry(radius, radius, np.linalg.norm(rB-rA), radialSegments=radialSegments)
  material = MeshPhongMaterial(color=color)
  mesh = Mesh(geometry,material)
  mesh.quaternion = quatUToV ([0,1,0], rB-rA)        # this is good
  #mesh.lookAt(tuple(rB-rA)) #mesh.rotateX(np.pi/2)  # these three.js commands cause minor problems with pythreejs
  mesh.position=tuple((rA+rB)/2)
  return mesh


#======================================================================
# txFont is transaprent
# txFont2 has black background
#=====================================================================  

img = PIL.Image.open('monospace.png')
xijc = np.array(img)
xijc = xijc[:,:,[0,0,0,0]]; xijc[:,:,:3]=[48,48,64];   # set font color by setting this triple
txFont = DataTexture(data=xijc, format='RGBAFormat', width=1024, height=1024)

img = PIL.Image.open('monospace.png')
xijc = np.array(img)
#xijc = xijc[:,:,[0,0,0,0]]; xijc[:,:,:3]=[99,99,99];   # set font color by setting this triple
txFont2 = DataTexture(data=xijc, format='RGBFormat', width=1024, height=1024)

def billboard (string='A', position=[0,0,0], rotation=[[1,0,0],[0,1,0],[0,0,1]], fontSize=1, ha='center', va='center', fontTexture=txFont):
  string = str(string)
  if string=='': return Group()
  position = np.array (position)
  rotation = np.array (rotation)
  glyphAsp = 2                  # fixed by font atlas
  glyphHei = fontSize
  glyphWid = glyphHei/glyphAsp
  a = glyphWid/2
  b = glyphHei/2
  k = .5/glyphAsp  
  material = MeshBasicMaterial(map=fontTexture, transparent=True)

  objects = []
  nmax = len(string)
  for n in range(nmax):
    #======== Where to draw character
    positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,1,0],[1,0,0],[1,1,0]])*[glyphWid,glyphHei,0] + [(2*n-nmax)*a,-b,0]
    positions = position + positions @ rotation
    #======== Which part of font atlas texture to use
    c = ord(string[n])
    i = c%16 ; j = c//16 ; iB = i+.5-k ; jB = j+.98 ; iE = i+.5+k ; jE = j+.02
    uv = (np.array([[0,0],[1,0],[0,1], [0,1],[1,0],[1,1]]) * [iE-iB,jE-jB] + [iB,jB] ) / 16
    geometry = BufferGeometry(attributes=dict(
      position= BufferAttribute (np.array(positions,dtype=np.float32)),   # must cast
      uv      = BufferAttribute (np.array(uv,       dtype=np.float32)))
    )
    #======== Add the glyph!
    mesh = Mesh(geometry,material)
    objects.append (mesh)
  group = Group (children=objects)
  # group.position = tuple(position @ rotation)  # we're not using this mechanism for positioning
  return group
  
# def zoomOut (positions, zoomFactor, center='auto'):
#   '''
#   zoomOut(...) takes a set of positions and scales each position about a center.
#   positions:  a two-dimensional array of coordinates (e.g., 10x3)
#   zoomFactor: a factor or array of factors (e.g., [1.1, 1.1, 1.1] to zoom out by 10%)
#   center:     center; if 'auto', use center-of-mass of positions
#   '''
#   if center=='auto':
#     center = np.mean(positions,axis=0)
#   return center + (positions - center)*zoomFactor

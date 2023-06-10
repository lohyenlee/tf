#========================
# Cute module
# Yen Lee Loh 2023-6-4
#========================
from IPython.display import display,HTML,Markdown
display(HTML('''
<style>
h1 { background-color: #AEA; padding: 0.8ex 0.8ex 0.5ex 0.8ex; border: 2px solid #8C8; }
h2 { background-color: #AEE; padding: 0.8ex 0.8ex 0.5ex 0.8ex; border: 2px solid #9CC; }
h3 { background-color: #EEA; padding: 0.8ex 0.8ex 0.5ex 0.8ex; border: 2px solid #CC9; }
</style>'''))
display(Markdown(r'''
$\newcommand{\mean}[1]{\langle #1 \rangle}$
$\newcommand{\bra}[1]{\langle #1 \rvert}$
$\newcommand{\ket}[1]{\lvert #1 \rangle}$
$\newcommand{\adag}{a^\dagger}$
$\newcommand{\mat}[1]{\underline{\underline{\mathbf{#1}}}}$
$\newcommand{\beq}{\qquad\begin{align}}$
$\newcommand{\eeq}{\end{align}}$
$\newcommand{\half}{\frac{1}{2}}$
'''))

import collections.abc as abc
import numpy as np; from numpy import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
rng = random.default_rng()

class simpleCallback:
  def _implements_train_batch_hooks(): return True
  def _implements_test_batch_hooks(): return True
  def _implements_predict_batch_hooks(): return True
  def on_train_begin(logs): return
  def on_train_end(logs): print();return
  def on_epoch_begin(epoch,logs): print ('\rEpoch =', epoch, end=''); return
  def on_epoch_end(epoch,logs): return
  def on_train_batch_begin(batch,logs): return
  def on_train_batch_end(batch,logs): return
  def set_model(a): return
  def set_params(a): return

def select (inputs, outputs, nT, nV, shuffle=False, seed=0):
  '''
  Split a dataset into training and validation sets.

  inputs:   numpy.ndarray, where inputs[n] is the nth input vector (which itself may be a multidimensional array)
  outputs:  numpy.ndarray, where outputs[n] is the nth output vector (which itself may be a multidimensional array)
  nT:       number of items in training set
  nV:       numebr of items in validation set
  '''
  nmax = len(inputs)
  assert len(outputs)==nmax and nT+nV<=nmax
  if shuffle:
    if seed>0:
      rng2 = np.random.default_rng (seed=seed)
      perm = rng2.permutation (nmax)
    else:
      perm = rng.permutation (nmax)
  else:
    perm = np.arange (nT+nV)
  indicesT = perm[:nT]
  indicesV = perm[-nV:]
  return inputs[indicesT], outputs[indicesT], inputs[indicesV], outputs[indicesV]

def gallery(xnij, cmap='viridis', labels=None, size=1, maxcols=20, wspace=0.02, hspace=0.02):  # size is in inches
  '''
  Display a row of images.

  xnij:     list of images (each image may be a PIL.Image or numpy.ndarray)  
  labels:   list of labels for image
  size:     width of each image, in inches
  maxcols:  maximum number of columns in gallery display before wrapping
  wspace:   fractional horizontal spacing between columns of images
  hspace:   fractional vertical spacing between rows of images
  '''  
  nmax = len(xnij)
  cols = min(nmax,maxcols) ; rows = (nmax+cols-1)//cols
  fig,axs = plt.subplots (rows,cols, figsize=(cols*size*(1+wspace),rows*size*(1+hspace)), gridspec_kw={'wspace':wspace,'hspace':hspace})
  if nmax==1: axs = np.array([[axs]])
  axs = axs.flatten()
  for ax in axs: ax.axis ('off')
  for n in range(nmax):
      ax = axs[n]
      if isinstance (cmap, abc.Iterable) and not isinstance (cmap, str): c = cmap[n]
      else: c = cmap
      ax.imshow (xnij[n], cmap=c)
      ax.set_aspect('equal')
      if isinstance (labels, abc.Iterable):
        ax.set_title (str(labels[n]))
        

def axgrid (widths=4, heights=2, ha=.5, va=.5, bottomtotop=False, labels=None, removeticks=True, padl=0, padt=0):
  '''
  Make a Figure and an array of Axes, arranged in a grid layout.
  
  Examples:
  
  >>> axgrid (3,1)                      # One plot of size 3x1
  >>> axgrid ([1,4,2,3], [1])           # One row of plots, all of height 1
  >>> axgrid (6, [.2, .4, .2])          # One column of plots
  >>> axgrid ([.2,3,3], [.2,.4,.4,.4])  # Grid with unequal widths and heights
  >>> axgrid ([.2,3,3], [.2,.4,.4,.4], bottomtotop=True) # Reverse vertical order of plots
  
  If *widths* and *heights* are both 2D arrays, some of the plots may be smaller than the allotted grid cell.
  In this case, *ha* and *va* determine horizontal alignment and vertical alignment.  For example:
  
  >>> axgrid ([[2,2,3],[2,3,2]], [[1,1,1],[2,1,2]], ha='left', va='top', labels='auto')
  
  Rows are usually in top-to-bottom order.  This may be reversed using the *bottomtotop* argument:
  
  >>> axgrid ([[2,2,3],[2,3,2]], [[1,1,1],[2,1,2]], ha='right', va='center', labels='auto', bottomtotop=True)
  
  In order to address the Figure and Axes objects, one should save the return values:
  
  >>> fig,axs = axgrid ([[2,2,3],[2,3,2]], [[1,1,1],[2,1,2]], removeticks=False)
  >>> ax = axs[0,0]; ax.plot ([1,2],[1,2])
  >>> ax = axs[1,2]; ax.plot ([1,2],[1,2]);
  
  The Axes in row i and column j is axs[i,j].  It has size width[i,j] x height[i,j].
  These conventions are consistent with matrix indexing conventions (and plt.subplots and numpy.ndarray)
  Generally, where indices are concerned, row indices are quoted before column indices.
  However, where physical dimensions are concerned, widths are quoted before heights,
  according to the conventional ordering of Cartesian coordinates (x,y) (and plt.plot).
  
  Parameters
  ----------
  widths, heights : scalar, 1D, or 2D array-like
        
  Returns
  ----------------
  fig, axs : Figure object and numpy.ndarray of Axes objects
  
  Other Parameters
  ----------------
  ha : 'left', 'center', 'right', float between 0 and 1; or 1D or 2D array of such specifications
  va : 'top', 'center', 'bottom', float between 0 and 1; or 1D 2D array of such specifications
  bottomtotop : False (default) or True
  labels : 
    None                  do not draw labels
    'auto'                label each Axes as axs[rowNumber,columnNumber]
    2D array of strings   custom labels to draw in the center of each Axes
  removeticks :
    True                  set each Axes to show only the frame (and no ticks)
    False                 leave Axes tick marks intact
  removeframe : 
    TBD
  '''
  #======== Determine number of grid cells
  wij = np.array (widths) 
  hij = np.array (heights)
  if wij.ndim==0: wij = np.array([wij])
  if hij.ndim==0: hij = np.array([hij])
  jmax = wij.shape[-1]
  imax = hij.shape[0]
  if wij.ndim==1: wij = np.tile (wij, (imax,1))                # Extend 1D to 2D
  if hij.ndim==0: hij = np.tile (hij, (imax,jmax))             # Extend 0D to 2D
  if hij.ndim==1: hij = np.tile (np.array([hij]).T, (1,jmax))  # Extend 1D to 2D
  assert hij.shape == wij.shape,'ERROR: axgrid was supplied with incompatible widths and heights!'
  if not bottomtotop:
    wij = np.flipud (wij)
    hij = np.flipud (hij)
  #======== Deal with padding
  plij = np.array (padl) # padding left
  ptij = np.array (padt)    # padding top
  if plij.ndim==0: plij = np.tile (plij, (imax,jmax))
  if ptij.ndim==0: ptij = np.tile (ptij, (imax,jmax))
  #======== Determine dimensions of grid cells
  wj = np.max (wij + plij, axis=0)
  hi = np.max (hij + ptij, axis=1)
  w = np.sum (wj)
  h = np.sum (hi)
  xj = np.concatenate ([[0], np.cumsum (wj)])
  yi = np.concatenate ([[0], np.cumsum (hi)])
  uij = np.array(ha)   # Array of horizontal alignment pars
  vij = np.array(va)   # Array of vertical alignment pars
  if uij.ndim==0: uij = np.tile (uij, (imax,jmax))
  if uij.ndim==1: uij = np.tile (uij, (imax,1))
  if vij.ndim==0: vij = np.tile (vij, (imax,jmax))
  if vij.ndim==1: vij = np.tile (np.array([vij]).T, (1,jmax))
  for i in range(imax):
    for j in range(jmax):
      if isinstance(uij[i,j],str): uij[i,j] = {'left':0, 'center':0.5, 'right':1}.get(uij[i,j])
      if isinstance(vij[i,j],str): vij[i,j] = {'top':0, 'center':0.5, 'bottom':1}.get(vij[i,j])
  uij = uij.astype (np.float64)
  vij = vij.astype (np.float64)
  #======== Create Axes
  fig,axs = plt.subplots (imax, jmax, figsize=(w,h))
  axs = np.array(axs).reshape ((imax,jmax))   # ensure this is always a imax*jmax numpy array of Axes
  for i in range(imax):
    for j in range(jmax):
      i2 = i if bottomtotop else imax-1-i
      x = (xj[j] + uij[i,j]*(wj[j] - wij[i,j])) / w
      y = (yi[i] + vij[i,j]*(hi[i] - hij[i,j])) / h
      axs[i2,j].set_position ([ x, y, wij[i,j]/w, hij[i,j]/h])
  if isinstance(labels,str) and labels=='auto':
    labels = np.array([[f'axs[{i},{j}]\n{wij[i][j]}x{hij[i][j]}' for j in range(jmax)] for i in range(imax)])
  if removeticks:
    for i in range(imax):
      for j in range(jmax):
        axs[i,j].set_xticks ([])
        axs[i,j].set_yticks ([])
  if isinstance(labels,np.ndarray):
    for i in range(imax):
      for j in range(jmax):
        axs[i,j].text (.5, .5, labels[i,j], ha='center', va='center', fontsize=20)
        axs[i,j].set_facecolor ('#FFFFCC')
  return fig,axs

def modifyAxSize (ax, wnew, hnew):
  wfig,hfig = ax.figure.get_size_inches()
  x0,y0,x1,y1 = ax.get_position().extents
  x0 *= wfig; x1 *= wfig; y0 *= hfig; y1 *= hfig
  x1 = x0 + wnew
  ym = (y0+y1)/2; y0=ym-hnew/2; y1=ym+hnew/2
  x0 /= wfig; x1 /= wfig; y0 /= hfig; y1 /= hfig
  ax.set_position([x0,y0,x1-x0,y1-y0])
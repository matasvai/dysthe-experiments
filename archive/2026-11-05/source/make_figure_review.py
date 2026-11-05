"""Standalone review gallery; scientific plots, not a new report or theorem test."""
import os,sys,json,ast
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.signal import resample
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from decay_dataset import setup,weighted
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output/figure_review';OUT.mkdir(exist_ok=True,parents=True)
# Import only the pure marching-tetrahedra function, without executing the older gallery.
tree=ast.parse((ROOT/'report/make_visual_gallery.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mesh');exec(compile(ast.Module(body=[node],type_ignores=[]),'<mesh>','exec'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'text.color':'#18384b','axes.labelcolor':'#18384b','axes.edgecolor':'#a8b5bf','grid.color':'#ccd5db','grid.alpha':.4,'savefig.facecolor':'white'})
TEAL='#168f9a';ORANGE='#d98542';PURPLE='#6858ac';RED='#b94a64'
items=[];data=[]
def save(fig,num,title,category,caption):
 name=f'{num:02d}_{title.lower().replace(" ","_").replace("/","_")}'
 fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',dpi=220)
 items.append(dict(number=num,title=title,category=category,caption=caption,png=name+'.png',svg=name+'.svg'));plt.close(fig)
def surface(ax,a,x,color):
 tri=mesh(a,.3*a.max(),x)
 n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-15)
 light=np.array([.3,-.5,.8]);light/=np.linalg.norm(light);shade=.42+.58*abs(n@light)
 fc=np.clip(np.array(to_rgb(color))[None,:]*shade[:,None]+.12,0,1)
 ax.add_collection3d(Poly3DCollection(tri,facecolors=fc,edgecolor='none',alpha=1,rasterized=True))
 ax.set(xlim=(-24,24),ylim=(-24,24),zlim=(-24,24),xlabel='$x$',ylabel='$y$',zlabel='$z$');ax.set_box_aspect((1,1,1));ax.view_init(22,-58)
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_ticks([-20,0,20]);axis.pane.fill=False
 ax.tick_params(labelsize=8,pad=0)
for j,(family,label,color) in enumerate([('random_11','Random seed 11',TEAL),('random_22','Random seed 22',ORANGE),('random_33','Random seed 33',PURPLE),('two_packets','Two interfering packets',RED),('anisotropic','Anisotropic packet',TEAL),('smooth_bump','Compact smooth packet',ORANGE)]):
 x,f,fh,w,K=setup(family,-1,0.,192,144);xx=x.numpy();sel=np.flatnonzero(abs(xx)<=24);xc=xx[sel];fields=[];centerlines=[]
 for t in [0,4,8]:
  u=f if t==0 else torch.fft.ifftn(fh*torch.exp(1j*t*w));a=u.abs().numpy();fields.append(u[:,96,:].numpy()[np.ix_(sel,sel)]);centerlines.append(u[:,96,96].numpy())
  crop=a[np.ix_(sel,sel,sel)];assert crop.max()==a.max()
  if t==0:fig=plt.figure(figsize=(13.5,4.8));fig.suptitle(label+' | three spatial coordinates',fontsize=17,y=.98)
  ax=fig.add_subplot(1,3,[0,4,8].index(t)+1,projection='3d');surface(ax,crop,xc,color);ax.set_title(f't = {t}  ·  peak {a.max():.3g}',fontsize=11,pad=12)
  data.append(dict(family=family,t=t,peak=float(a.max()),l2=float(torch.linalg.vector_norm(u)*.75**1.5),surface_crop_boundary_ratio=float(max(crop[0].max(),crop[-1].max(),crop[:,0].max(),crop[:,-1].max(),crop[:,:,0].max(),crop[:,:,-1].max())/a.max())))
 fig.subplots_adjust(left=.02,right=.98,bottom=.06,top=.85,wspace=.02)
 save(fig,2*j+1,label+' volumes','Wave shapes','Homogeneous canonical solution, three spatial axes. Surfaces show 30% of each snapshot’s own peak; thresholds vary with time, so these panels compare shape. Printed values compare amplitude. Grid 192³, length 144, spacing 0.75; qualitative views, not a new convergence audit.')
 fig,axs=plt.subplots(2,3,figsize=(13,7));peak=max(abs(v).max() for v in fields)
 for col,(t,v) in enumerate(zip([0,4,8],fields)):
  im=axs[0,col].imshow(abs(v).T/peak,origin='lower',extent=[-24.375,24.375]*2,cmap='magma',vmin=0,vmax=1);axs[0,col].set(title=f't = {t}',xlabel='$x$',ylabel='$z$ at $y=0$')
  xf=np.arange(1152)*.125-72;keep=abs(xf)<=24;line=resample(centerlines[col],1152)[keep];xf=xf[keep];env=abs(line)/peak;carrier=np.real(line*np.exp(3j*xf))/peak
  axs[1,col].fill_between(xf,-env,env,color=color,alpha=.12);axs[1,col].plot(xf,carrier,color='#18384b',lw=1);axs[1,col].plot(xf,env,color=color,lw=1.8);axs[1,col].plot(xf,-env,color=color,lw=1.2)
  axs[1,col].set(xlim=(-24,24),ylim=(-1.05,1.05),xlabel='$x$ at $y=z=0$',ylabel='Relative slice amplitude');axs[1,col].grid(alpha=.25)
 fig.subplots_adjust(left=.07,right=.87,bottom=.09,top=.87,wspace=.28,hspace=.38);cax=fig.add_axes([.9,.57,.014,.27]);fig.colorbar(im,cax=cax,label='Magnitude / shared slice maximum')
 fig.suptitle(label+' | amplitude and visible oscillations',fontsize=17)
 save(fig,2*j+2,label+' slices and waves','Wave shapes','Top: envelope slices with one fixed color scale across all three times. Bottom: Re{S(t)f exp(3ix)} and ±|S(t)f| on the centerline, using that same divisor. The complex centerline is Fourier-interpolated to spacing 0.125 for drawing. The extra carrier is illustrative, not inferred from physical water data. A centerline can miss the full-volume maximum.')
 del f,fh,w,K,u,a,crop
 print('Rendered',family,flush=True)
R=ROOT/'results/research_experiments'
s=json.loads((R/'sweep.json').read_text());follow=json.loads((R/'followup.json').read_text());fix={(a['name'],a['t']):a for a in follow}
def ratio(a):return fix.get((a['name'],a['t']),{}).get('refined_ratio',a['ratio'])
fig,axs=plt.subplots(2,2,figsize=(11,7),sharey=True)
for row,reg in enumerate(['homogeneous','inhomogeneous']):
 for col,sx in enumerate([3,6]):
  ax=axs[row,col]
  for t,c in zip([4,8,16],[PURPLE,ORANGE,TEAL]):
   rr=sorted([a for a in s if a['regime']==reg and a['sx']==sx and a['group']=='axial sweep' and a['t']==t],key=lambda a:a['offset'])
   ax.plot([a['offset'] for a in rr],[1e3*ratio(a) for a in rr],'o-',c=c,label=f't = {t}',lw=2)
   for a in rr:
    if (a['name'],a['t']) in fix:ax.scatter(a['offset'],1e3*ratio(a),marker='D',s=60,color=c,edgecolor='white',zorder=4)
  ax.axvline(0,color='gray',ls=':',lw=1);ax.set(title=f'{reg.capitalize()} · width {sx}',xlabel=r'Frequency offset $q_x+\beta$',ylabel='R(t) × 1000');ax.grid();ax.legend(fontsize=9)
fig.tight_layout();save(fig,13,'Frequency sweep across three times','Decay results','Twenty Gaussian profiles. Circles pass the original 2% domain/grid screen; diamonds use the independent 0.1% follow-up. Connecting lines guide the eye between sampled frequencies; they are not an exhaustive spectral search.')
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
for col,reg in enumerate(['homogeneous','inhomogeneous']):
 for name,label,c in [(reg+'_cancel','Zero integral',ORANGE),('homogeneous_sx3_offset0' if col==0 else 'inhomogeneous_unshifted_control','Gaussian control',TEAL)]:
  rr=[a for a in s if a['name']==name];axs[col].plot([a['t'] for a in rr],[1e3*ratio(a) for a in rr],'o-',c=c,label=label,lw=2.3)
 axs[col].set(title=reg.capitalize(),xlabel='Time t',ylabel='R(t) × 1000',xticks=[4,8,16]);axs[col].legend();axs[col].grid()
fig.tight_layout();save(fig,14,'Does cancellation change the curve','Decay results','The odd Gaussian has exactly zero continuum integral. Curves use L¹ normalization. The inhomogeneous cancellation curve is nearly flat from 8 to 16 within tested numerical differences; this is not evidence of a new asymptotic exponent.')
a=json.loads((R/'identity.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for reg,c in [('homogeneous',TEAL),('inhomogeneous',ORANGE)]:
 rr=[v for v in a if v['regime']==reg];axs[0].plot([v['width'] for v in rr],[1000*v['ratio'] for v in rr],'o-',label=reg,c=c,lw=2)
 axs[1].plot([v['width'] for v in rr],[100*v['domain_change'] for v in rr],'o-',label=reg,c=c,lw=2)
for ax in axs:ax.set_xlabel('Gaussian width');ax.grid();ax.legend()
axs[0].set(title='Concentrating the initial field',ylabel='R(1) × 1000');axs[1].set(title='Change after expanding the search',ylabel='Relative peak difference (%)');fig.tight_layout()
save(fig,15,'Narrow inputs and their reliability','Decay results','Ten exactly L¹-normalized Gaussian tests at t=1. The domain comparison also enlarges the radial search. Width 0.75 is the smallest tested; these values do not determine the optimal kernel norm.')
fig,ax=plt.subplots(figsize=(11,6));labels=[v['name'].replace('inhomogeneous_','').replace('_',' ')+f" / t={v['t']}" for v in follow];y=np.arange(len(follow))
ax.barh(y-.17,[100*v['original_to_refined'] for v in follow],height=.3,color=ORANGE,label='Correction to original fine-grid peak');ax.barh(y+.17,[100*v['domain_and_radial_change'] for v in follow],height=.3,color=TEAL,label='Change between follow-up searches');ax.set(yticks=y,yticklabels=labels,xlabel='Relative difference (%)',title='Why the flagged cases needed follow-up');ax.invert_yaxis();ax.legend();ax.grid(axis='x');fig.tight_layout()
save(fig,16,'Corrections after numerical refinement','Reliability','All eight flagged measurements were followed up. Orange bars compare original fine-grid values with refined integral candidates. Teal bars compare the two later broad searches and are much smaller. Their small size is not a certified global error bound.')
a=json.loads((R/'quadrature.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for (reg,fam),c in zip(dict.fromkeys((v['regime'],v['family']) for v in a),[TEAL,PURPLE,ORANGE,RED]):
 rr=[v for v in a if v['regime']==reg and v['family']==fam];label=reg.split('_')[0]+' / '+('near' if fam=='near_degeneracy' else 'Gaussian')
 for ax,key in zip(axs,['relative_disagreement','order_change']):ax.semilogy([v['t'] for v in rr],[v[key] for v in rr],'o-',c=c,label=label)
for ax in axs:ax.set(xlabel='Time t at fixed coordinates',ylabel='Relative complex-field difference',xticks=[256,512,1024]);ax.grid();ax.legend(fontsize=8)
axs[0].set_title('Independent implementations');axs[1].set_title('256 versus 512 Gauss nodes per panel');fig.tight_layout();save(fig,17,'Quadrature agreement and order sensitivity','Reliability','Twelve evaluations at the previously identified t=256 peak coordinates. Later times are pointwise stress tests, not new spatial maxima. Order sensitivity becomes visible at 1024 even when the higher-order rule agrees with the adaptive calculation.')
a=json.loads((ROOT/'results/long_time/summary.json').read_text());latest=json.loads((ROOT/'results/spatial_reliability/summary.json').read_text());fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for ax,c in zip(axs.flat,latest):
 rr=[v for v in a if v['regime']==c['regime'] and v['family']==c['family']];ax.plot([v['t'] for v in rr],[1000*v['ratio'] for v in rr],'o-',color=TEAL,lw=2)
 ax.scatter(256,1000*c['continuum_candidate']['ratio'],marker='D',color=ORANGE,s=55,zorder=4)
 ax.set(xscale='log',xticks=[16,32,64,128,256],xticklabels=['16','32','64','128','256'],xlabel='Time t',ylabel='R(t) × 1000',title=c['regime'].split('_')[0].capitalize()+' / '+c['family'].replace('_',' '));ax.grid()
fig.tight_layout();save(fig,18,'The longest checked decay window','Decay results','Original long-time curve with the more strongly checked t=256 endpoint overlaid as an orange diamond. A rising ratio is compatible with a bounded inequality. No t=512 or 1024 global maximum is plotted.')
fig,axs=plt.subplots(1,2,figsize=(11,4.6));x=np.linspace(-2,1.5,350);y=np.linspace(-2,2,350);X,Y=np.meshgrid(x,y)
for ax,sg,b,title in zip(axs,[-1,1],[0,1],['Homogeneous','Inhomogeneous']):
 det=(sg*X+b)*(X*(sg*X+b)-Y*Y);ax.contour(X,Y,det,levels=[0],colors=TEAL,linewidths=2)
 ax.axvline(-b,c=ORANGE,ls='--',label='Multiplier cusp');ax.scatter([-b+d for d in [-.75,-.25,0,.25,.75]],[0]*5,c=PURPLE,label='Axial centers',zorder=3)
 if b:ax.scatter([.5]*3,[np.sqrt(.75)+d for d in [-.2,0,.2]],marker='D',c=RED,label='Curvature probes')
 ax.set(title=title,xlabel='$k_x$',ylabel='$k_y$ at $k_z=0$',xlim=(-2,1.5),ylim=(-2,2));ax.set_aspect('equal');ax.grid();ax.legend(fontsize=8,loc='lower left')
fig.tight_layout();save(fig,19,'Where we placed the test spectra','Experiment design','Teal curves mark det(D²w)=0 on kz=0; dashed orange lines mark the multiplier cusp. Points are centers of Gaussian spectra with nonzero bandwidth, not delta functions or strict support constraints. The homogeneous degeneracy and cusp coincide.')
a=json.loads((R/'scaling.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for fam,c in [('gaussian',TEAL),('smooth_bump',ORANGE),('random_33',PURPLE)]:
 rr=[v for v in a if v['family']==fam];axs[0].plot([v['t'] for v in rr],[1000*v['ratio'] for v in rr],'o-',c=c,label=fam.replace('_',' '))
axs[0].set(xscale='log',xlabel='Time t with correspondingly rescaled input',ylabel='R(t) × 1000',title='One exact scaling identity');axs[0].legend();axs[0].grid()
fams=list(dict.fromkeys(v['family'] for v in a));axs[1].barh([f.replace('_',' ') for f in fams],[1e16*max(v['relative_field_error'] for v in a if v['family']==f) for f in fams],color=TEAL);axs[1].invert_yaxis();axs[1].set(title='All eleven recipes',xlabel='Relative field discrepancy × $10^{16}$');fig.tight_layout();save(fig,20,'Scaling as a solver check','Reliability','Thirty-three evaluations of the homogeneous scaling identity on correspondingly rescaled periodic grids. Agreement checks implementation consistency, not whole-space convergence. The nearly constant lines correspond to different rescaled inputs, not one fixed input evolving in time.')
(OUT/'catalog.json').write_text(json.dumps(items,indent=2));(OUT/'new_snapshot_diagnostics.json').write_text(json.dumps(data,indent=2));print('Saved 20 figures',flush=True)

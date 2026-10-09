# Model Alpha V-Phaselet X1.2 — optimized static renderer for Pyodide.
# Renderer only: no ART_CODE payload and no CLI.
# Usage: render_art_code(payload, "/tmp/result.png", size=256, supersample=1)
# Dependencies: numpy, scipy, pillow.

from pathlib import Path
import math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter, distance_transform_edt, map_coordinates


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.maximum(np.asarray(c, dtype=np.float32), 0)
    return np.where(c <= .0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - .055)


def hexrgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return np.array([int(h[i:i+2], 16) for i in (0,2,4)], np.float32) / 255.0


def aces(x):
    x = np.maximum(x, 0)
    return np.clip((x * (2.51*x + .03)) / (x * (2.43*x + .59) + .14), 0, 1)


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-7)


def idw(points, values, q, power=2.0):
    points = np.asarray(points, np.float32)
    values = np.asarray(values, np.float32)
    q = np.asarray(q, np.float32)
    d2 = ((q[:,None,:] - points[None,:,:])**2).sum(-1)
    w = 1 / np.maximum(d2, 1e-6) ** (power/2)
    if values.ndim == 1:
        return (w @ values) / np.maximum(w.sum(1), 1e-7)
    return (w @ values) / np.maximum(w.sum(1)[:,None], 1e-7)


class SparseField:
    def __init__(self, pins, channels=1, linear_color=False):
        self.channels = channels
        self.linear_color = linear_color
        if not pins:
            self.points = np.array([[0.5,0.5]], np.float32)
            self.values = np.zeros((1,channels), np.float32)
            self.interp = None
            return
        pts, vals = [], []
        for pin in pins:
            x,y,*v = pin
            pts.append((float(x),float(y)))
            if linear_color:
                if len(v) == 1 and isinstance(v[0], str):
                    vv = srgb_to_linear(hexrgb(v[0]))
                else:
                    vv = srgb_to_linear(np.asarray(v[:3], np.float32))
            else:
                vv = np.asarray(v[:channels], np.float32)
            vals.append(np.atleast_1d(vv))
        self.points = np.asarray(pts, np.float32)
        self.values = np.asarray(vals, np.float32)
        self.interp = None
        if len(self.points) >= 3:
            try:
                self.interp = CloughTocher2DInterpolator(self.points, self.values)
            except Exception:
                self.interp = None

    def __call__(self, q):
        q = np.asarray(q, np.float32)
        if len(self.points) == 1:
            return np.broadcast_to(self.values[0], (len(q), self.values.shape[1])).copy()
        if self.interp is not None:
            out = np.asarray(self.interp(q[:,0], q[:,1]), np.float32)
            if out.ndim == 1:
                out = out[:,None]
            bad = ~np.isfinite(out).all(1)
            if bad.any():
                out[bad] = idw(self.points, self.values, q[bad])
            return out
        out = idw(self.points, self.values, q)
        if out.ndim == 1:
            out = out[:,None]
        return out


class LocalTriField:
    """C0 local Delaunay field with barycentric interpolation.

    This preserves local authored contrast instead of globally smoothing every
    sparse pin. Optional bounds receive nearest-color corner extension.
    """
    def __init__(self,pins,channels=1,linear_color=False,bounds=None):
        pts=[]; vals=[]
        for pin in pins:
            x,y,*v=pin
            pts.append((float(x),float(y)))
            if linear_color:
                if len(v)==1 and isinstance(v[0],str):
                    vv=srgb_to_linear(hexrgb(v[0]))
                else:
                    vv=srgb_to_linear(np.asarray(v[:3],np.float32))
            else:
                vv=np.asarray(v[:channels],np.float32)
            vals.append(np.atleast_1d(vv))
        self.points=np.asarray(pts,np.float32)
        self.values=np.asarray(vals,np.float32)
        if bounds is not None and len(self.points):
            x0,y0,x1,y1=bounds
            corners=np.asarray([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
            cv=[]
            for q in corners:
                j=np.argmin(((self.points-q)**2).sum(1))
                cv.append(self.values[j])
            self.points=np.vstack([self.points,corners])
            self.values=np.vstack([self.values,np.asarray(cv)])
        self.tri=None
        if len(self.points)>=3:
            try:self.tri=Delaunay(self.points)
            except Exception:self.tri=None

    def __call__(self,q):
        q=np.asarray(q,np.float32)
        if len(self.points)==1:
            return np.broadcast_to(self.values[0],(len(q),self.values.shape[1])).copy()
        if self.tri is None:
            out=idw(self.points,self.values,q,power=2.0)
            return out[:,None] if out.ndim==1 else out
        s=self.tri.find_simplex(q)
        out=np.empty((len(q),self.values.shape[1]),np.float32)
        good=s>=0
        if np.any(good):
            T=self.tri.transform[s[good],:2]
            r=q[good]-self.tri.transform[s[good],2]
            bc=np.einsum("nij,nj->ni",T,r)
            w=np.c_[bc,1-bc.sum(1)]
            verts=self.tri.simplices[s[good]]
            out[good]=(self.values[verts]*w[:,:,None]).sum(1)
        if np.any(~good):
            out[~good]=idw(self.points,self.values,q[~good],power=2.0)
        return out


class SmoothScalarField:
    """Regularized sparse scalar reconstruction for relief/material channels.

    Clough-Tocher is excellent for RGB painting but can turn sparse height pins
    into saddle/diamond artifacts. Linear RBF gives a smoother global relief
    field while retaining sparse authoring.
    """
    def __init__(self, pins, smoothing=7e-4):
        pts=[]; vals=[]
        for x,y,v,*_ in pins:
            pts.append((float(x),float(y)))
            vals.append(float(v))
        self.points=np.asarray(pts,np.float32)
        self.values=np.asarray(vals,np.float32)[:,None]
        self.rbf=None
        if len(self.points)>=2:
            try:
                self.rbf=RBFInterpolator(
                    self.points,self.values,
                    kernel="linear",
                    smoothing=float(smoothing)
                )
            except Exception:
                self.rbf=None

    def __call__(self,q):
        q=np.asarray(q,np.float32)
        if len(self.points)==1:
            return np.full((len(q),1),self.values[0,0],np.float32)
        if self.rbf is not None:
            out=np.asarray(self.rbf(q),np.float32)
            if out.ndim==1: out=out[:,None]
            return out
        out=idw(self.points,self.values,q,power=1.45)
        if out.ndim==1: out=out[:,None]
        return out


class OrientationField:
    """Sparse angle reconstruction via interpolated unit vectors."""
    def __init__(self,pins):
        pts=[]; vals=[]
        for x,y,a,*_ in pins:
            pts.append((float(x),float(y)))
            vals.append((math.cos(float(a)),math.sin(float(a))))
        self.field=SparseField([(x,y,v[0],v[1]) for (x,y),v in zip(pts,vals)],2)

    def __call__(self,q):
        v=self.field(q)
        n=np.linalg.norm(v,axis=1,keepdims=True)
        v=v/np.maximum(n,1e-7)
        return np.arctan2(v[:,1],v[:,0])[:,None]


def sample_path(commands, curve_steps=18):
    """Sample a compact Canvas/SVG-like path command list to polygon/polyline points.

    Commands:
      ("M", x, y)
      ("L", x, y)
      ("C", c1x, c1y, c2x, c2y, x, y)
      ("Q", cx, cy, x, y)
      ("Z",)
    """
    if not commands:
        return []
    # Plain (x,y) points remain valid.
    if not isinstance(commands[0][0], str):
        return [(float(x),float(y)) for x,y in commands]

    out=[]; start=None; cur=None
    for cmd in commands:
        op=cmd[0].upper()
        if op=="M":
            cur=np.array([cmd[1],cmd[2]],float); start=cur.copy(); out.append(tuple(cur))
        elif op=="L":
            cur=np.array([cmd[1],cmd[2]],float); out.append(tuple(cur))
        elif op=="C":
            if cur is None: raise ValueError("C requires a current point")
            p0=cur.copy(); p1=np.array(cmd[1:3],float); p2=np.array(cmd[3:5],float); p3=np.array(cmd[5:7],float)
            for t in np.linspace(0,1,max(3,int(curve_steps))+1)[1:]:
                q=(1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3
                out.append(tuple(q))
            cur=p3
        elif op=="Q":
            if cur is None: raise ValueError("Q requires a current point")
            p0=cur.copy(); p1=np.array(cmd[1:3],float); p2=np.array(cmd[3:5],float)
            for t in np.linspace(0,1,max(3,int(curve_steps))+1)[1:]:
                q=(1-t)**2*p0+2*(1-t)*t*p1+t*t*p2
                out.append(tuple(q))
            cur=p2
        elif op=="Z":
            if start is not None and (not out or np.linalg.norm(np.array(out[-1])-start)>1e-9):
                out.append(tuple(start))
            cur=start.copy() if start is not None else cur
        else:
            raise ValueError(f"Unknown path op {op}")
    return out


def tapered_ribbon(points, width0, width1=None):
    """Convert a center polyline into a variable-width polygon."""
    pts=np.asarray(points,float)
    if len(pts)<2: return [tuple(p) for p in pts]
    if width1 is None: width1=width0
    tang=np.zeros_like(pts)
    tang[0]=pts[1]-pts[0]; tang[-1]=pts[-1]-pts[-2]
    if len(pts)>2: tang[1:-1]=pts[2:]-pts[:-2]
    n=np.linalg.norm(tang,axis=1,keepdims=True); tang=tang/np.maximum(n,1e-9)
    nor=np.c_[-tang[:,1],tang[:,0]]
    t=np.linspace(0,1,len(pts))
    w=(float(width0)*(1-t)+float(width1)*t)*.5
    left=pts+nor*w[:,None]; right=pts-nor*w[:,None]
    poly=np.vstack([left,right[::-1]])
    return [tuple(p) for p in poly]


class AlphaCanvas:
    """
    Model Alpha v0 diagnostic renderer.

    Canonical per-layer state:
      coverage, linear RGB, height/direct-normal, roughness,
      reflectance, emission, optional transmission approximation,
      orientation metadata, depth/order.

    Authoring concepts:
      patch(...)  sparse properties inside an authoritative coverage mask
      stroke(...) thin directional structure
      field(...)  direct image-space arrays/functions
    """
    def __init__(self, size=192, supersample=3, seed=4):
        self.out = int(size)
        self.ss = int(supersample)
        self.S = self.out * self.ss
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.hdr = np.zeros((self.S,self.S,3), np.float32)
        self.alpha = np.zeros((self.S,self.S), np.float32)
        self.depth = np.full((self.S,self.S), -1e9, np.float32)
        yy,xx = np.mgrid[:self.S,:self.S].astype(np.float32)
        self.u = (xx + .5) / self.S
        self.v = (yy + .5) / self.S
        self.stats = dict(patches=0, strokes=0, fields=0)

    def background(self, top="#101622", bottom="#29364a"):
        a = srgb_to_linear(hexrgb(top))
        b = srgb_to_linear(hexrgb(bottom))
        t = self.v[...,None]
        self.hdr = a*(1-t) + b*t
        self.alpha[:] = 1

    def _mask_polygon(self, boundary, smooth=False):
        boundary = sample_path(boundary) if boundary and isinstance(boundary[0][0], str) else boundary
        pts = np.asarray(boundary, np.float32)
        if smooth and len(pts) >= 4:
            # Chaikin: stable and non-overshooting.
            for _ in range(3):
                q = np.roll(pts,-1,axis=0)
                pts = np.stack([.75*pts+.25*q, .25*pts+.75*q],axis=1).reshape(-1,2)
        im = Image.new("L",(self.S,self.S),0)
        d = ImageDraw.Draw(im)
        d.polygon([(float(x)*self.S,float(y)*self.S) for x,y in pts], fill=255)
        return np.asarray(im,np.float32)/255.0

    def _environment(self, R, roughness):
        # Neutral sky/ground + broad studio softboxes. HDR.
        ry = np.clip(R[...,1]*.5+.5, 0, 1)
        sky = np.array([.08,.14,.24],np.float32)*(1-ry[...,None]) + np.array([.36,.52,.72],np.float32)*ry[...,None]
        env = sky.copy()

        def softbox(direction, sharpness, color, gain):
            d = np.asarray(direction,np.float32); d /= np.linalg.norm(d)
            dot = np.clip((R*d).sum(-1),0,1)
            # Roughness broadens and dims coherent reflections.
            k = np.maximum(2.0, sharpness*(1-.78*np.clip(roughness,0,1)))
            return (dot**k)[...,None] * np.asarray(color,np.float32) * gain

        env += softbox((-0.68,.45,.58), 55, (1.0,.96,.90), 3.8)
        env += softbox(( .72,.30,.62), 70, (1.0,.24,.55), 2.8)
        env += softbox((-.72,.18,.67), 80, (.30,.72,1.0), 3.0)
        env += softbox((0.0,.85,.30), 25, (1.0,1.0,.92), 1.5)
        return env

    def _shade(self, color, normal, rough, refl, emission, occlusion=1.0, orientation=None, anisotropy=None):
        N = normalize(normal)
        V = np.zeros_like(N); V[...,2] = 1
        L1 = np.array([-.42,-.58,.70],np.float32)
        L2 = np.array([ .58,-.20,.79],np.float32)
        L1 /= np.linalg.norm(L1); L2 /= np.linalg.norm(L2)

        nd1 = np.clip(N@L1,0,1)
        nd2 = np.clip(N@L2,0,1)
        ambient = np.array([.18,.22,.29],np.float32)
        diff = ambient + nd1[...,None]*np.array([1.0,.90,.78])*.80 + nd2[...,None]*np.array([.32,.53,.88])*.28

        # Reflection direction for fixed frontal view.
        nz=N[...,2]
        R=np.stack([2*nz*N[...,0], 2*nz*N[...,1], 2*nz*nz-1],-1)
        env=self._environment(R, rough)

        f0=np.clip(refl,0,1)[...,None]
        F=f0 + (1-f0)*(1-np.clip(nz,0,1))[...,None]**5

        # Compact GGX-like highlight.
        H = normalize(L1 + np.array([0,0,1],np.float32))
        nh=np.clip(N@H,0,1)
        rr=np.maximum(.04,rough)
        a=rr*rr
        D=(a*a)/(np.pi*(nh*nh*(a*a-1)+1)**2+1e-6)
        spec=(D*np.clip(refl,0,1)*.22)[...,None]*np.array([1.0,.96,.90])

        occ=np.asarray(occlusion,np.float32)
        if occ.ndim==0: occ=np.full(color.shape[:2],occ,np.float32)
        out = color*diff*(1-F)*occ[...,None] + env*F*occ[...,None] + spec + emission

        # Orientation-aware anisotropic highlight. Orientation is a general
        # screen-space tangent field, not a semantic material type.
        if orientation is not None and anisotropy is not None:
            th=np.asarray(orientation,np.float32)
            an=np.clip(np.asarray(anisotropy,np.float32),0,1)
            tx=np.cos(th); ty=np.sin(th)
            bx=-ty; by=tx
            H2=normalize(L2 + np.array([0,0,1],np.float32))
            ht=H2[0]*tx + H2[1]*ty
            hb=H2[0]*bx + H2[1]*by
            hz=max(float(H2[2]),.08)
            at=np.maximum(.025, rough*(1.0+2.4*an))
            ab=np.maximum(.025, rough*(1.0-.78*an))
            lobe=np.exp(-((ht/at)**2 + (hb/ab)**2)/(hz*hz))
            an_spec=(lobe*refl*(.25+1.4*an))[...,None]*np.array([.72,.86,1.0],np.float32)
            out += an_spec
        return np.maximum(out,0)

    def _composite(self, rgb, a, depth=0.0):
        a=np.clip(a,0,1)
        take = a > 1e-5
        # Ordered layer compositing; depth is scalar in v0.
        visible = take & (depth >= self.depth)
        aa = np.where(visible,a,0)
        self.hdr = self.hdr*(1-aa[...,None]) + rgb*aa[...,None]
        self.alpha = self.alpha + aa*(1-self.alpha)
        self.depth = np.where(visible, depth, self.depth)

    def patch(self, boundary, color_pins, *, height_pins=None, normal_pins=None,
              roughness_pins=None, reflectance_pins=None, emission_pins=None,
              orientation_pins=None, anisotropy_pins=None, color_continuity=1, height_continuity=1,
              alpha=1.0, smooth=False, depth=0.0, transmission=0.0,
              ior=1.45, absorption=None, thickness=.5, unlit=False,
              lighting_mix=1.0, glow=0.0, blur=0.0, shadow=None, blend="normal",
              background_sample_shift=(0,0)):
        mask=self._mask_polygon(boundary,smooth)*float(alpha)
        yy,xx=np.where(mask>1e-5)
        if not len(xx): return
        q=np.c_[(xx+.5)/self.S,(yy+.5)/self.S]

        bounds=(float(xx.min()+.5)/self.S,float(yy.min()+.5)/self.S,
                float(xx.max()+.5)/self.S,float(yy.max()+.5)/self.S)
        if int(color_continuity)==0:
            C=LocalTriField(color_pins,3,linear_color=True,bounds=bounds)(q).reshape(-1,3)
        else:
            C=SparseField(color_pins,3,linear_color=True)(q).reshape(-1,3)
        C=np.clip(C,0,1)

        if height_pins:
            if int(height_continuity)==0:
                hf=LocalTriField(height_pins,1,linear_color=False,bounds=bounds)
            else:
                hf=SmoothScalarField(height_pins)
            H=np.zeros((self.S,self.S),np.float32)
            H[yy,xx]=hf(q)[:,0]
            H=gaussian_filter(H,1.1*self.ss)
            gy,gx=np.gradient(H)
            N=np.zeros((self.S,self.S,3),np.float32)
            N[...,0]=-gx*self.S*1.35
            N[...,1]=-gy*self.S*1.35
            N[...,2]=1
        else:
            N=np.zeros((self.S,self.S,3),np.float32); N[...,2]=1

        if normal_pins:
            nf=SparseField(normal_pins,2)
            nv=nf(q)
            N[yy,xx,0]+=nv[:,0]
            N[yy,xx,1]+=nv[:,1]
            xy=np.clip(N[yy,xx,:2],-.98,.98)
            N[yy,xx,2]=np.sqrt(np.maximum(.02,1-(xy*xy).sum(1)))
        N=normalize(N)

        def scalar_field(pins, default):
            if not pins: return np.full(len(q),default,np.float32)
            return SparseField(pins,1)(q)[:,0]

        rough=np.clip(scalar_field(roughness_pins,.45),.02,1)
        refl=np.clip(scalar_field(reflectance_pins,.05),0,1)
        if emission_pins:
            E=SparseField(emission_pins,3,linear_color=False)(q)
        else:
            E=np.zeros((len(q),3),np.float32)

        color_img=np.zeros((self.S,self.S,3),np.float32)
        rough_img=np.ones((self.S,self.S),np.float32)*.45
        refl_img=np.zeros((self.S,self.S),np.float32)
        emit_img=np.zeros((self.S,self.S,3),np.float32)
        ori_img=np.zeros((self.S,self.S),np.float32)
        aniso_img=np.zeros((self.S,self.S),np.float32)
        color_img[yy,xx]=C; rough_img[yy,xx]=rough; refl_img[yy,xx]=refl; emit_img[yy,xx]=E

        if orientation_pins:
            ori=OrientationField(orientation_pins)(q)[:,0]
            ori_img[yy,xx]=ori
        if anisotropy_pins:
            aniso=scalar_field(anisotropy_pins,0)
            aniso_img[yy,xx]=np.clip(aniso,0,1)

        base_unlit=color_img+emit_img
        if unlit:
            shaded=base_unlit
        else:
            lit=self._shade(
                color_img,N,rough_img,refl_img,np.zeros_like(emit_img),
                orientation=ori_img if orientation_pins else None,
                anisotropy=aniso_img if anisotropy_pins else None
            )+emit_img
            lm=float(np.clip(lighting_mix,0,1))
            shaded=base_unlit*(1-lm)+lit*lm

        # General 2D refraction approximation: the local normal displaces the
        # already-rendered background; IOR controls bend and absorption tints it.
        if transmission>0:
            eta=1/max(float(ior),1.001)
            strength=(1-eta)*(.045+.055*float(thickness))
            shiftx=N[...,0]*self.S*strength + float(background_sample_shift[0])*self.S
            shifty=N[...,1]*self.S*strength + float(background_sample_shift[1])*self.S
            gy,gx=np.mgrid[:self.S,:self.S].astype(np.float32)
            coords=[np.clip(gy+shifty,0,self.S-1),np.clip(gx+shiftx,0,self.S-1)]
            transmitted=np.stack([
                map_coordinates(self.hdr[...,k],coords,order=1,mode="nearest")
                for k in range(3)
            ],-1)
            if absorption is not None:
                absorb=np.asarray(absorption,np.float32)
                transmitted*=np.exp(-absorb[None,None,:]*float(thickness))
            # Keep a reflective surface term while transmitting the background.
            shaded=shaded*(1-transmission*.78)+transmitted*transmission

        if blur and float(blur)>0:
            sig=max(.5,float(blur)*self.S)
            prem=gaussian_filter(shaded*mask[...,None],(sig,sig,0))
            bm=gaussian_filter(mask,sig)
            shaded=np.divide(prem,bm[...,None],out=np.zeros_like(prem),where=bm[...,None]>1e-7)
            mask=np.clip(bm,0,1)

        # Optional generic drop/contact shadow, expressed in normalized image units.
        if shadow is not None:
            dx,dy,sigma,strength,*rest=shadow
            scol=np.array([0.0,0.0,0.0],np.float32)
            if rest:
                rv=rest[0]
                scol=srgb_to_linear(hexrgb(rv)) if isinstance(rv,str) else np.asarray(rv,np.float32)
            sm=gaussian_filter(mask,max(.5,float(sigma)*self.S))
            sm=np.roll(sm,(int(float(dy)*self.S),int(float(dx)*self.S)),(0,1))
            sa=np.clip(sm*float(strength),0,1)
            self.hdr=self.hdr*(1-sa[...,None])+scol[None,None,:]*sa[...,None]

        if blend=="normal":
            self._composite(shaded,mask,depth)
        else:
            visible=(mask>1e-5)&(depth>=self.depth)
            aa=np.where(visible,mask,0).astype(np.float32)
            if blend=="add":
                self.hdr += shaded*aa[...,None]
            elif blend=="screen":
                dst=np.clip(self.hdr,0,1); src=np.clip(shaded,0,1)
                mixed=1-(1-dst)*(1-src)
                self.hdr=self.hdr*(1-aa[...,None])+mixed*aa[...,None]
            elif blend=="multiply":
                mixed=self.hdr*np.clip(shaded,0,1)
                self.hdr=self.hdr*(1-aa[...,None])+mixed*aa[...,None]
            else:
                raise ValueError("blend must be normal, add, screen or multiply")
            self.alpha=self.alpha+aa*(1-self.alpha)
            self.depth=np.where(visible,depth,self.depth)

        # Generic optical glow after the source is present.
        if glow and float(glow)>0:
            prem=shaded*mask[...,None]
            self.hdr += gaussian_filter(prem,(float(glow)*self.S,float(glow)*self.S,0))*.60

        self.stats["patches"]+=1

    def stroke(self, points, width, color="#ffffff", alpha=1.0, depth=0.0, blur=0.0, emission=0.0):
        # Local-bounds rasterization. Same authoring semantics as before, but
        # avoids allocating/compositing a full S×S array for every tiny line.
        logical = sample_path(points, curve_steps=20) if points and isinstance(points[0][0], str) else [(float(x),float(y)) for x,y in points]
        px=[(float(x)*self.S,float(y)*self.S) for x,y in logical]
        if len(px)<2:
            return
        tapered = isinstance(width,(tuple,list,np.ndarray))
        if tapered:
            w0=float(width[0]); w1=float(width[-1])
            maxw=max(w0,w1)
        else:
            w0=w1=float(width); maxw=float(width)
        radius=max(1.0,maxw*self.S*.5)
        margin=radius+max(2.0,float(blur)*self.S*4.0)+2.0
        xs=[p[0] for p in px]; ys=[p[1] for p in px]
        x0=max(0,int(math.floor(min(xs)-margin))); x1=min(self.S,int(math.ceil(max(xs)+margin))+1)
        y0=max(0,int(math.floor(min(ys)-margin))); y1=min(self.S,int(math.ceil(max(ys)+margin))+1)
        if x1<=x0 or y1<=y0:
            return
        im=Image.new("L",(x1-x0,y1-y0),0)
        d=ImageDraw.Draw(im)
        local=[(x-x0,y-y0) for x,y in px]
        if tapered:
            # Segmentwise tapered rasterization with round joints.
            for j in range(len(local)-1):
                t=(j+.5)/max(1,len(local)-1)
                ww=(w0*(1-t)+w1*t)*self.S
                p0=local[j]; p1=local[j+1]
                d.line([p0,p1],fill=255,width=max(1,int(ww)))
                rr=max(.5,ww*.5)
                d.ellipse((p1[0]-rr,p1[1]-rr,p1[0]+rr,p1[1]+rr),fill=255)
        else:
            d.line(local,fill=255,width=max(1,int(float(width)*self.S)),joint="curve")
        if blur>0:
            im=im.filter(ImageFilter.GaussianBlur(blur*self.S))
        a=np.asarray(im,np.float32)/255*alpha
        if not np.any(a>1e-5):
            return
        c=srgb_to_linear(hexrgb(color))
        rgb=np.broadcast_to(c,(y1-y0,x1-x0,3)).copy()
        if emission>0:
            rgb += c*emission

        depth_view=self.depth[y0:y1,x0:x1]
        visible=(a>1e-5)&(depth>=depth_view)
        aa=np.where(visible,a,0).astype(np.float32)
        hdr=self.hdr[y0:y1,x0:x1]
        hdr[:]=hdr*(1-aa[...,None])+rgb*aa[...,None]
        av=self.alpha[y0:y1,x0:x1]
        av[:]=av+aa*(1-av)
        depth_view[:]=np.where(visible,depth,depth_view)
        self.stats["strokes"]+=1

    def field(self, rgb=None, coverage=None, depth=0.0, mode="over", emission=None):
        cov=np.ones((self.S,self.S),np.float32) if coverage is None else np.clip(np.asarray(coverage,np.float32),0,1)
        if callable(rgb): col=np.asarray(rgb(self.u,self.v),np.float32)
        else: col=np.asarray(rgb,np.float32)
        if col.ndim==1: col=np.broadcast_to(col,(self.S,self.S,3)).copy()
        if emission is not None: col=col+np.asarray(emission,np.float32)
        if mode=="add":
            self.hdr += col*cov[...,None]
        elif mode=="screen":
            # HDR-friendly approximation to screen: additive brightening.
            self.hdr += np.maximum(col,0)*cov[...,None]
        else:
            self._composite(col,cov,depth)
        self.stats["fields"]+=1

    def contact_ao(self, upper_mask, lower_mask, radius=.022, strength=.55, offset=(0,.012)):
        # Local projected contact field. Distance falloff avoids the giant
        # blurred-mask blob from v0 while preserving grounding around contact.
        um=np.asarray(upper_mask,np.float32)
        lm=np.asarray(lower_mask,np.float32)
        shifted=np.roll(um,(int(offset[1]*self.S),int(offset[0]*self.S)),(0,1))
        dist=distance_transform_edt(shifted<.08)/self.S
        fall=np.exp(-(dist/max(radius,1e-5))**2)
        ao=np.clip(fall*strength,0,.78)*lm
        self.hdr *= (1-ao[...,None])
        return ao

    def oriented_texture(self, base_mask, orientation_fn, *, seed=1,
                         flow_length=.030, samples=11, frequency=1.0,
                         strength=.08, color=(1,1,1), depth=0.21, mode="add"):
        """Line-integral-convolution-like stochastic texture.

        One generic orientation field creates coherent elongated detail without
        requiring literal authored hairs. It can drive fur, fabric, feathers,
        brushed metal, wood grain, etc.
        """
        mask=np.asarray(base_mask,np.float32)
        theta=np.zeros((self.S,self.S),np.float32)
        # orientation_fn is authored at logical normalized coordinates.
        # Accept a full field, a scalar constant, or fall back to scalar evaluation.
        try:
            candidate=np.asarray(orientation_fn(self.u,self.v),np.float32)
            if candidate.shape==mask.shape:
                theta=candidate
            elif candidate.ndim==0:
                theta.fill(float(candidate))
            else:
                raise ValueError
        except Exception:
            theta=np.zeros((self.S,self.S),np.float32)
            for yy in range(self.S):
                for xx in range(self.S):
                    theta[yy,xx]=float(orientation_fn((xx+.5)/self.S,(yy+.5)/self.S))

        rng=np.random.default_rng(seed)
        base=rng.normal(0,1,(self.S,self.S)).astype(np.float32)
        # add a lower-frequency seed so the texture has meso + micro scale
        base=.72*base+.28*gaussian_filter(base,1.2*self.ss)

        gy,gx=np.mgrid[:self.S,:self.S].astype(np.float32)
        acc=np.zeros_like(base); wsum=0.0
        half=max(1,samples//2)
        for k in range(-half,half+1):
            t=k/max(half,1)*flow_length*self.S
            xs=gx+np.cos(theta)*t
            ys=gy+np.sin(theta)*t
            samp=map_coordinates(base,[ys,xs],order=1,mode="reflect")
            w=1-abs(k)/(half+1)
            acc+=samp*w; wsum+=w
        lic=acc/max(wsum,1e-6)
        # remove very-low-frequency bias; normalize inside the material.
        lic=lic-gaussian_filter(lic,max(1,flow_length*self.S*.55))
        vals=lic[mask>.12]
        sd=float(vals.std()) if vals.size else 1.0
        lic/=max(sd,1e-6)

        col=np.asarray(color,np.float32)
        if col.shape==(3,):
            residual=lic[...,None]*col[None,None,:]*float(strength)*float(frequency)
        else:
            residual=lic[...,None]*float(strength)*float(frequency)
        self.field(residual,mask,depth=depth,mode=mode)
        return lic

    def oriented_fibers(self, base_mask, orientation_fn, *, color="#d8d8d8",
                        density=1.0, length=.040, width=.0020, jitter=.45,
                        alpha=.75, seed=1, depth=0.2, extend=.012,
                        edge_only=False, edge_band=.018):
        # Generic mesostructure: thousands of short strokes driven by a tangent field.
        mask=np.asarray(base_mask,np.float32)
        r=np.random.default_rng(seed)
        target=int(self.out*self.out*density*.018)
        if edge_only:
            inside=distance_transform_edt(mask>.15)/self.S
            seed_mask=(mask>.15)&(inside<edge_band)
        else:
            seed_mask=mask>.15
        im=Image.new("L",(self.S,self.S),0)
        d=ImageDraw.Draw(im)
        for _ in range(target):
            for __ in range(10):
                x,y=r.random(),r.random()
                iy=min(self.S-1,int(y*self.S)); ix=min(self.S-1,int(x*self.S))
                if seed_mask[iy,ix]: break
            else: continue
            ang=float(orientation_fn(x,y))
            ang += r.normal(0,jitter*.35)
            L=length*(.55+r.random()*.9)
            # let fibers reach beyond the base mask naturally
            x0=x-math.cos(ang)*L*.35; y0=y-math.sin(ang)*L*.35
            x1=x+math.cos(ang)*(L+extend)*.65; y1=y+math.sin(ang)*(L+extend)*.65
            val=int(255*alpha*(.55+.45*r.random()))
            d.line((x0*self.S,y0*self.S,x1*self.S,y1*self.S),fill=val,
                   width=max(1,int(width*self.S)))
        a=np.asarray(im,np.float32)/255
        c=srgb_to_linear(hexrgb(color))
        rgb=np.broadcast_to(c,(self.S,self.S,3)).copy()
        # Fibers are allowed outside the base mask by design.
        self._composite(rgb,a,depth)
        return a

    def finish(self, bloom=.10, vignette=.08, grain=.002, tone_map="aces"):
        hdr=np.maximum(self.hdr,0)
        if bloom>0:
            bright=np.maximum(hdr-1,0)
            hdr=hdr+gaussian_filter(bright,(3*self.ss,3*self.ss,0))*bloom
        if tone_map=="aces":
            out=aces(hdr)
        elif tone_map=="none":
            out=np.clip(hdr,0,1)
        else:
            raise ValueError("tone_map must be 'aces' or 'none'")
        if vignette:
            r2=((self.u-.5)**2+(self.v-.5)**2)*2
            out*=1-vignette*np.clip(r2,0,1)[...,None]
        out=linear_to_srgb(out)
        if grain:
            out += self.rng.normal(0,grain,out.shape)
        out=np.clip(out,0,1)
        # area-integrate supersampled radiance/display samples
        out=out.reshape(self.out,self.ss,self.out,self.ss,3).mean((1,3))
        return Image.fromarray(np.uint8(np.clip(out,0,1)*255+.5))


def rect(x0,y0,x1,y1):
    return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]


def ellipse_poly(cx,cy,rx,ry,n=80):
    return [(cx+math.cos(t)*rx,cy+math.sin(t)*ry) for t in np.linspace(0,math.tau,n,endpoint=False)]


def mask_from_poly(size, ss, pts, smooth=False):
    S=size*ss
    p=np.asarray(pts,np.float32)
    if smooth and len(p)>=4:
        for _ in range(3):
            q=np.roll(p,-1,axis=0)
            p=np.stack([.75*p+.25*q,.25*p+.75*q],axis=1).reshape(-1,2)
    im=Image.new("L",(S,S),0); d=ImageDraw.Draw(im)
    d.polygon([(x*S,y*S) for x,y in p],fill=255)
    return np.asarray(im,np.float32)/255



class SharedSurface:
    """Deferred/shared surface for coherent geometry with authoritative internal regions.

    Build one coherent surface, paint internal albedo/material regions into it,
    then shade/composite once. This prevents subregions from becoming sticker-like
    mini-surfaces unless the author explicitly makes them separate surfaces.
    """
    def __init__(self, canvas, boundary, color_pins, *,
                 height_pins=None, normal_pins=None,
                 roughness_pins=None, reflectance_pins=None, emission_pins=None,
                 orientation_pins=None, anisotropy_pins=None,
                 color_continuity=1, height_continuity=1,
                 alpha=1.0, smooth=False, depth=0.0,
                 transmission=0.0, ior=1.45, absorption=None, thickness=.5):
        self.c=canvas
        self.depth=float(depth)
        self.transmission=float(transmission)
        self.ior=float(ior)
        self.absorption=absorption
        self.thickness=float(thickness)

        self.mask=canvas._mask_polygon(boundary,smooth)*float(alpha)
        S=canvas.S
        self.color=np.zeros((S,S,3),np.float32)
        self.rough=np.ones((S,S),np.float32)*.45
        self.refl=np.zeros((S,S),np.float32)
        self.emit=np.zeros((S,S,3),np.float32)
        self.ori=np.zeros((S,S),np.float32)
        self.aniso=np.zeros((S,S),np.float32)
        self.N=np.zeros((S,S,3),np.float32); self.N[...,2]=1

        yy,xx=np.where(self.mask>1e-5)
        self.yy,self.xx=yy,xx
        if not len(xx):
            return
        q=np.c_[(xx+.5)/S,(yy+.5)/S]
        bounds=(float(xx.min()+.5)/S,float(yy.min()+.5)/S,
                float(xx.max()+.5)/S,float(yy.max()+.5)/S)

        if int(color_continuity)==0:
            C=LocalTriField(color_pins,3,linear_color=True,bounds=bounds)(q)
        else:
            C=SparseField(color_pins,3,linear_color=True)(q)
        self.color[yy,xx]=np.clip(C,0,1)

        def scalar(pins,default):
            if not pins: return np.full(len(q),default,np.float32)
            return SparseField(pins,1)(q)[:,0]

        self.rough[yy,xx]=np.clip(scalar(roughness_pins,.45),.02,1)
        self.refl[yy,xx]=np.clip(scalar(reflectance_pins,.05),0,1)

        if emission_pins:
            self.emit[yy,xx]=SparseField(emission_pins,3)(q)

        if height_pins:
            if int(height_continuity)==0:
                hf=LocalTriField(height_pins,1,linear_color=False,bounds=bounds)
            else:
                hf=SmoothScalarField(height_pins)
            H=np.zeros((S,S),np.float32)
            H[yy,xx]=hf(q)[:,0]
            H=gaussian_filter(H,1.1*canvas.ss)
            gy,gx=np.gradient(H)
            self.N[...,0]=-gx*S*1.35
            self.N[...,1]=-gy*S*1.35
            self.N[...,2]=1

        if normal_pins:
            nv=SparseField(normal_pins,2)(q)
            self.N[yy,xx,0]+=nv[:,0]
            self.N[yy,xx,1]+=nv[:,1]
            xy=np.clip(self.N[yy,xx,:2],-.98,.98)
            self.N[yy,xx,2]=np.sqrt(np.maximum(.02,1-(xy*xy).sum(1)))
        self.N=normalize(self.N)

        if orientation_pins:
            self.ori[yy,xx]=OrientationField(orientation_pins)(q)[:,0]
        if anisotropy_pins:
            self.aniso[yy,xx]=np.clip(scalar(anisotropy_pins,0),0,1)

    def paint(self, boundary, color_pins=None, *, alpha=1.0, smooth=False,
              color_continuity=1, roughness_pins=None, reflectance_pins=None,
              emission_pins=None, orientation_pins=None, anisotropy_pins=None,
              normal_pins=None, mode="replace"):
        """Modify channels inside an internal region while retaining shared geometry."""
        c=self.c; S=c.S
        m=c._mask_polygon(boundary,smooth)*float(alpha)*self.mask
        yy,xx=np.where(m>1e-5)
        if not len(xx): return self
        q=np.c_[(xx+.5)/S,(yy+.5)/S]
        a=m[yy,xx,None]
        bounds=(float(xx.min()+.5)/S,float(yy.min()+.5)/S,
                float(xx.max()+.5)/S,float(yy.max()+.5)/S)

        if color_pins:
            if int(color_continuity)==0:
                C=LocalTriField(color_pins,3,linear_color=True,bounds=bounds)(q)
            else:
                C=SparseField(color_pins,3,linear_color=True)(q)
            C=np.clip(C,0,1)
            if mode=="replace":
                self.color[yy,xx]=self.color[yy,xx]*(1-a)+C*a
            elif mode=="multiply":
                mixed=self.color[yy,xx]*C
                self.color[yy,xx]=self.color[yy,xx]*(1-a)+mixed*a
            elif mode=="add":
                self.color[yy,xx]=np.clip(self.color[yy,xx]+C*a,0,1)
            else:
                raise ValueError("paint mode must be replace/multiply/add")

        def apply_scalar(target,pins,lo,hi):
            if not pins: return
            v=np.clip(SparseField(pins,1)(q)[:,0],lo,hi)
            target[yy,xx]=target[yy,xx]*(1-m[yy,xx])+v*m[yy,xx]

        apply_scalar(self.rough,roughness_pins,.02,1)
        apply_scalar(self.refl,reflectance_pins,0,1)
        apply_scalar(self.aniso,anisotropy_pins,0,1)

        if emission_pins:
            E=SparseField(emission_pins,3)(q)
            self.emit[yy,xx]=self.emit[yy,xx]*(1-a)+E*a

        if orientation_pins:
            O=OrientationField(orientation_pins)(q)[:,0]
            self.ori[yy,xx]=self.ori[yy,xx]*(1-m[yy,xx])+O*m[yy,xx]

        if normal_pins:
            nv=SparseField(normal_pins,2)(q)
            cur=self.N[yy,xx].copy()
            cur[:,0]=cur[:,0]*(1-m[yy,xx])+nv[:,0]*m[yy,xx]
            cur[:,1]=cur[:,1]*(1-m[yy,xx])+nv[:,1]*m[yy,xx]
            xy=np.clip(cur[:,:2],-.98,.98)
            cur[:,2]=np.sqrt(np.maximum(.02,1-(xy*xy).sum(1)))
            self.N[yy,xx]=normalize(cur)

        return self

    def relief_field(self, height, *, strength=1.0, smooth=0.0, mask=None):
        """Add arbitrary image-space microrelief to the shared normal field."""
        c=self.c
        if callable(height):
            H=np.asarray(height(c.u,c.v),np.float32)
        else:
            H=np.asarray(height,np.float32)
        if H.ndim==0:
            H=np.full((c.S,c.S),float(H),np.float32)
        if H.shape!=(c.S,c.S):
            raise ValueError("relief_field must resolve to SxS")
        m=self.mask if mask is None else self.mask*np.asarray(mask,np.float32)
        H=H*m
        if smooth and float(smooth)>0:
            H=gaussian_filter(H,max(.3,float(smooth)*c.S))
        gy,gx=np.gradient(H)
        self.N[...,0]+= -gx*c.S*float(strength)
        self.N[...,1]+= -gy*c.S*float(strength)
        self.N=normalize(self.N)
        return self

    def oriented_relief(self, orientation_fn, *, seed=1, flow_length=.020,
                        samples=11, strength=.020, detail_scale=1.0, mask=None):
        """Generic anisotropic stochastic relief driven by a tangent field."""
        c=self.c
        m=self.mask if mask is None else self.mask*np.asarray(mask,np.float32)
        try:
            th=np.asarray(orientation_fn(c.u,c.v),np.float32)
            if th.ndim==0:
                th=np.full((c.S,c.S),float(th),np.float32)
            if th.shape!=(c.S,c.S):
                raise ValueError
        except Exception:
            th=np.zeros((c.S,c.S),np.float32)
            for yy in range(c.S):
                for xx in range(c.S):
                    th[yy,xx]=float(orientation_fn((xx+.5)/c.S,(yy+.5)/c.S))

        rng=np.random.default_rng(seed)
        base=rng.normal(0,1,(c.S,c.S)).astype(np.float32)
        # Give the source both micro and meso frequency energy.
        base=.72*base+.28*gaussian_filter(base,max(.45,1.1*c.ss))
        gy,gx=np.mgrid[:c.S,:c.S].astype(np.float32)
        acc=np.zeros_like(base); wsum=0.0
        half=max(1,int(samples)//2)
        for k in range(-half,half+1):
            tt=k/max(half,1)*float(flow_length)*c.S
            xs=gx+np.cos(th)*tt
            ys=gy+np.sin(th)*tt
            samp=map_coordinates(base,[ys,xs],order=1,mode="reflect")
            w=1-abs(k)/(half+1)
            acc+=samp*w; wsum+=w
        lic=acc/max(wsum,1e-6)
        lic-=gaussian_filter(lic,max(.5,float(flow_length)*c.S*.55))
        vals=lic[m>.12]
        sd=float(vals.std()) if vals.size else 1.0
        lic/=max(sd,1e-6)
        return self.relief_field(lic*float(detail_scale),strength=float(strength),mask=m)

    def noise_relief(self, *, seed=1, sigma=.002, strength=.010, mask=None):
        c=self.c
        rng=np.random.default_rng(seed)
        n=rng.normal(0,1,(c.S,c.S)).astype(np.float32)
        if sigma and float(sigma)>0:
            n=gaussian_filter(n,max(.3,float(sigma)*c.S))
        vals=n[self.mask>.12]
        n=(n-(float(vals.mean()) if vals.size else 0))/(float(vals.std())+1e-6 if vals.size else 1)
        return self.relief_field(n,strength=float(strength),mask=mask)

    def render(self, *, lighting_mix=1.0, unlit=False, blur=0.0,
               glow=0.0, shadow=None, blend="normal",
               background_sample_shift=(0,0)):
        c=self.c
        base=self.color+self.emit
        if unlit:
            shaded=base.copy()
        else:
            lit=c._shade(self.color,self.N,self.rough,self.refl,np.zeros_like(self.emit),
                         orientation=self.ori,anisotropy=self.aniso)+self.emit
            lm=float(np.clip(lighting_mix,0,1))
            shaded=base*(1-lm)+lit*lm

        # shared-surface transmission uses final shared normal field
        if self.transmission>0:
            eta=1/max(self.ior,1.001)
            strength=(1-eta)*(.045+.055*self.thickness)
            shiftx=self.N[...,0]*c.S*strength+float(background_sample_shift[0])*c.S
            shifty=self.N[...,1]*c.S*strength+float(background_sample_shift[1])*c.S
            gy,gx=np.mgrid[:c.S,:c.S].astype(np.float32)
            coords=[np.clip(gy+shifty,0,c.S-1),np.clip(gx+shiftx,0,c.S-1)]
            transmitted=np.stack([
                map_coordinates(c.hdr[...,k],coords,order=1,mode="nearest")
                for k in range(3)
            ],-1)
            if self.absorption is not None:
                absorb=np.asarray(self.absorption,np.float32)
                transmitted*=np.exp(-absorb[None,None,:]*self.thickness)
            shaded=shaded*(1-self.transmission*.78)+transmitted*self.transmission

        mask=self.mask.copy()
        if blur and float(blur)>0:
            sig=max(.5,float(blur)*c.S)
            prem=gaussian_filter(shaded*mask[...,None],(sig,sig,0))
            bm=gaussian_filter(mask,sig)
            shaded=np.divide(prem,bm[...,None],out=np.zeros_like(prem),where=bm[...,None]>1e-7)
            mask=np.clip(bm,0,1)

        if shadow is not None:
            dx,dy,sigma,strength,*rest=shadow
            scol=np.array([0,0,0],np.float32)
            if rest:
                rv=rest[0]
                scol=srgb_to_linear(hexrgb(rv)) if isinstance(rv,str) else np.asarray(rv,np.float32)
            sm=gaussian_filter(mask,max(.5,float(sigma)*c.S))
            sm=np.roll(sm,(int(float(dy)*c.S),int(float(dx)*c.S)),(0,1))
            sa=np.clip(sm*float(strength),0,1)
            c.hdr=c.hdr*(1-sa[...,None])+scol[None,None,:]*sa[...,None]

        if blend=="normal":
            c._composite(shaded,mask,self.depth)
        else:
            visible=(mask>1e-5)&(self.depth>=c.depth)
            aa=np.where(visible,mask,0).astype(np.float32)
            if blend=="add":
                c.hdr+=shaded*aa[...,None]
            elif blend=="screen":
                dst=np.clip(c.hdr,0,1); src=np.clip(shaded,0,1)
                mixed=1-(1-dst)*(1-src)
                c.hdr=c.hdr*(1-aa[...,None])+mixed*aa[...,None]
            elif blend=="multiply":
                mixed=c.hdr*np.clip(shaded,0,1)
                c.hdr=c.hdr*(1-aa[...,None])+mixed*aa[...,None]
            else:
                raise ValueError("blend must be normal/add/screen/multiply")
            c.alpha=c.alpha+aa*(1-c.alpha)
            c.depth=np.where(visible,self.depth,c.depth)

        if glow and float(glow)>0:
            prem=shaded*mask[...,None]
            c.hdr+=gaussian_filter(prem,(float(glow)*c.S,float(glow)*c.S,0))*.60

        c.stats["patches"]+=1
        return self


def _shared_surface(self,*args,**kwargs):
    return SharedSurface(self,*args,**kwargs)

AlphaCanvas.surface=_shared_surface


class CompoundSharedSurface(SharedSurface):
    """SharedSurface initialized from an arbitrary already-unioned coverage mask."""
    def __init__(self, canvas, mask, color_pins, *,
                 height_pins=None, normal_pins=None,
                 roughness_pins=None, reflectance_pins=None, emission_pins=None,
                 orientation_pins=None, anisotropy_pins=None,
                 color_continuity=1, height_continuity=1,
                 alpha=1.0, depth=0.0,
                 transmission=0.0, ior=1.45, absorption=None, thickness=.5):
        self.c=canvas
        self.depth=float(depth)
        self.transmission=float(transmission)
        self.ior=float(ior)
        self.absorption=absorption
        self.thickness=float(thickness)
        self.mask=np.clip(np.asarray(mask,np.float32)*float(alpha),0,1)

        S=canvas.S
        self.color=np.zeros((S,S,3),np.float32)
        self.rough=np.ones((S,S),np.float32)*.45
        self.refl=np.zeros((S,S),np.float32)
        self.emit=np.zeros((S,S,3),np.float32)
        self.ori=np.zeros((S,S),np.float32)
        self.aniso=np.zeros((S,S),np.float32)
        self.N=np.zeros((S,S,3),np.float32); self.N[...,2]=1

        yy,xx=np.where(self.mask>1e-5)
        self.yy,self.xx=yy,xx
        if not len(xx): return
        q=np.c_[(xx+.5)/S,(yy+.5)/S]
        bounds=(float(xx.min()+.5)/S,float(yy.min()+.5)/S,
                float(xx.max()+.5)/S,float(yy.max()+.5)/S)

        if int(color_continuity)==0:
            C=LocalTriField(color_pins,3,linear_color=True,bounds=bounds)(q)
        else:
            C=SparseField(color_pins,3,linear_color=True)(q)
        self.color[yy,xx]=np.clip(C,0,1)

        def scalar(pins,default):
            if not pins:return np.full(len(q),default,np.float32)
            return SparseField(pins,1)(q)[:,0]

        self.rough[yy,xx]=np.clip(scalar(roughness_pins,.45),.02,1)
        self.refl[yy,xx]=np.clip(scalar(reflectance_pins,.05),0,1)

        if emission_pins:
            self.emit[yy,xx]=SparseField(emission_pins,3)(q)

        if height_pins:
            hf=LocalTriField(height_pins,1,linear_color=False,bounds=bounds) if int(height_continuity)==0 else SmoothScalarField(height_pins)
            H=np.zeros((S,S),np.float32); H[yy,xx]=hf(q)[:,0]
            H*=self.mask
            H=gaussian_filter(H,1.1*canvas.ss)
            gy,gx=np.gradient(H)
            self.N[...,0]=-gx*S*1.35
            self.N[...,1]=-gy*S*1.35
            self.N[...,2]=1

        if normal_pins:
            nv=SparseField(normal_pins,2)(q)
            self.N[yy,xx,0]+=nv[:,0]
            self.N[yy,xx,1]+=nv[:,1]
            xy=np.clip(self.N[yy,xx,:2],-.98,.98)
            self.N[yy,xx,2]=np.sqrt(np.maximum(.02,1-(xy*xy).sum(1)))
        self.N=normalize(self.N)

        if orientation_pins:
            self.ori[yy,xx]=OrientationField(orientation_pins)(q)[:,0]
        if anisotropy_pins:
            self.aniso[yy,xx]=np.clip(scalar(anisotropy_pins,0),0,1)


class CompoundBuilder:
    """Union multiple generic paths/strokes into one authoritative shared surface."""
    def __init__(self,group):
        self.g=group
        self.c=group.c
        self.mask=np.zeros((self.c.S,self.c.S),np.float32)

    def add_path(self,boundary,smooth=False,group=None):
        G=self.g if group is None else group
        pts=G.path(boundary)
        m=self.c._mask_polygon(pts,smooth)
        self.mask=np.maximum(self.mask,m)
        return self

    def add_stroke(self,points,width,group=None):
        G=self.g if group is None else group
        logical=G.path(points)
        if logical and isinstance(logical[0][0],str):
            logical=sample_path(logical,curve_steps=24)
        S=self.c.S
        im=Image.new("L",(S,S),0); d=ImageDraw.Draw(im)
        px=[(float(x)*S,float(y)*S) for x,y in logical]
        if len(px)<2:return self
        sx,sy=G._scale(); sc=math.sqrt(max(sx*sy,1e-12))
        tapered=isinstance(width,(tuple,list,np.ndarray))
        if tapered:
            w0=float(width[0])*sc; w1=float(width[-1])*sc
            for j in range(len(px)-1):
                t=(j+.5)/max(1,len(px)-1)
                ww=(w0*(1-t)+w1*t)*S
                p0,p1=px[j],px[j+1]
                d.line([p0,p1],fill=255,width=max(1,int(ww)))
                rr=max(.5,ww*.5)
                d.ellipse((p1[0]-rr,p1[1]-rr,p1[0]+rr,p1[1]+rr),fill=255)
        else:
            ww=float(width)*sc*S
            d.line(px,fill=255,width=max(1,int(ww)),joint="curve")
        self.mask=np.maximum(self.mask,np.asarray(im,np.float32)/255)
        return self


    def add_sweep(self,points,widths,group=None,round_caps=True):
        """Union a variable-width swept path into this compound surface.

        widths may be a scalar or a sequence sampled uniformly along arc length.
        This is generic centerline + profile geometry with no semantic knowledge.
        """
        G=self.g if group is None else group
        logical=G.path(points)
        if logical and isinstance(logical[0][0],str):
            logical=sample_path(logical,curve_steps=32)
        if len(logical)<2:
            return self

        P=np.asarray(logical,np.float32)
        seg=np.linalg.norm(P[1:]-P[:-1],axis=1)
        cum=np.r_[0,np.cumsum(seg)]
        total=max(float(cum[-1]),1e-8)
        t=cum/total

        sx,sy=G._scale(); sc=math.sqrt(max(sx*sy,1e-12))
        if np.isscalar(widths):
            W=np.full(len(P),float(widths)*sc,np.float32)
        else:
            ws=np.asarray(widths,np.float32).ravel()
            W=np.interp(t,np.linspace(0,1,len(ws)),ws)*sc

        T=np.zeros_like(P)
        T[1:-1]=P[2:]-P[:-2]
        T[0]=P[1]-P[0]; T[-1]=P[-1]-P[-2]
        L=np.maximum(np.linalg.norm(T,axis=1,keepdims=True),1e-8)
        T=T/L
        N=np.c_[-T[:,1],T[:,0]]
        left=P+N*(W[:,None]*.5)
        right=P-N*(W[:,None]*.5)
        poly=np.vstack([left,right[::-1]])

        S=self.c.S; MS=2*S
        im=Image.new("L",(MS,MS),0); d=ImageDraw.Draw(im)
        d.polygon([(float(x)*MS,float(y)*MS) for x,y in poly],fill=255)
        if round_caps:
            for q,w in ((P[0],W[0]),(P[-1],W[-1])):
                rr=float(w)*MS*.5; x=float(q[0])*MS; y=float(q[1])*MS
                d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=255)
        m=np.asarray(im.resize((S,S),Image.Resampling.LANCZOS),np.float32)/255.0
        self.mask=np.maximum(self.mask,m)
        return self

    def surface(self,color_pins,**kw):
        depth=self.g.depth_offset+float(kw.pop("depth",0.0))
        for key,kind in (
            ("height_pins","scalar"),("normal_pins","normal"),
            ("roughness_pins","scalar"),("reflectance_pins","scalar"),
            ("emission_pins","scalar"),("orientation_pins","orientation"),
            ("anisotropy_pins","scalar")
        ):
            if kw.get(key) is not None:
                kw[key]=self.g._pins(kw[key],kind)
        surf=CompoundSharedSurface(self.c,self.mask,self.g._pins(color_pins),depth=depth,**kw)
        return LocalSurface(self.g,surf)


class LocalGroup:
    """Generic hierarchical 2D local coordinate frame.

    Local coordinates are normally 0..1. A group only provides spatial
    organization; it has no semantic knowledge of faces, dogs, gears, etc.
    """
    def __init__(self, canvas, matrix=None, depth_offset=0.0):
        self.c=canvas
        self.M=np.eye(3,dtype=np.float32) if matrix is None else np.asarray(matrix,np.float32)
        self.depth_offset=float(depth_offset)

    @staticmethod
    def _local_matrix(x=0.0,y=0.0,w=1.0,h=1.0,rotation=0.0,anchor=(0.0,0.0),skew=(0.0,0.0)):
        ax,ay=anchor
        ca,sa=math.cos(rotation),math.sin(rotation)
        kx,ky=math.tan(skew[0]),math.tan(skew[1])
        T0=np.array([[1,0,-ax],[0,1,-ay],[0,0,1]],np.float32)
        S=np.array([[w,0,0],[0,h,0],[0,0,1]],np.float32)
        K=np.array([[1,kx,0],[ky,1,0],[0,0,1]],np.float32)
        R=np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]],np.float32)
        T1=np.array([[1,0,x],[0,1,y],[0,0,1]],np.float32)
        return T1@R@K@S@T0

    def compound(self):
        return CompoundBuilder(self)

    def group(self,x=0.0,y=0.0,w=1.0,h=1.0,rotation=0.0,anchor=(0.0,0.0),skew=(0.0,0.0),depth=0.0):
        return LocalGroup(self.c,self.M@self._local_matrix(x,y,w,h,rotation,anchor,skew),
                          self.depth_offset+float(depth))

    def _xy(self,x,y):
        p=self.M@np.array([float(x),float(y),1.0],np.float32)
        return float(p[0]),float(p[1])

    def _linear(self):
        return self.M[:2,:2].astype(np.float32)

    def _scale(self):
        A=self._linear()
        sx=float(np.linalg.norm(A[:,0])); sy=float(np.linalg.norm(A[:,1]))
        return sx,sy

    def _angle(self,a):
        A=self._linear()
        v=A@np.array([math.cos(float(a)),math.sin(float(a))],np.float32)
        return math.atan2(float(v[1]),float(v[0]))

    def _normal(self,nx,ny):
        A=self._linear()
        try:
            v=np.linalg.inv(A).T@np.array([float(nx),float(ny)],np.float32)
        except Exception:
            v=np.array([float(nx),float(ny)],np.float32)
        return float(v[0]),float(v[1])

    def path(self,boundary):
        if not boundary:return boundary
        if isinstance(boundary[0][0],str):
            out=[]
            for cmd in boundary:
                op=cmd[0].upper()
                if op=="Z":
                    out.append(("Z",)); continue
                nums=list(cmd[1:])
                vals=[op]
                for i in range(0,len(nums),2):
                    vals.extend(self._xy(nums[i],nums[i+1]))
                out.append(tuple(vals))
            return out
        return [self._xy(x,y) for x,y in boundary]

    def _pins(self,pins,kind="scalar"):
        if not pins:return pins
        out=[]
        for p in pins:
            x,y,*rest=p
            gx,gy=self._xy(x,y)
            if kind=="normal":
                nx,ny=self._normal(rest[0],rest[1])
                out.append((gx,gy,nx,ny,*rest[2:]))
            elif kind=="orientation":
                out.append((gx,gy,self._angle(rest[0]),*rest[1:]))
            else:
                out.append((gx,gy,*rest))
        return out

    def patch(self,boundary,color_pins,**kw):
        kw["depth"]=self.depth_offset+float(kw.get("depth",0.0))
        for key,kind in (
            ("height_pins","scalar"),("normal_pins","normal"),
            ("roughness_pins","scalar"),("reflectance_pins","scalar"),
            ("emission_pins","scalar"),("orientation_pins","orientation"),
            ("anisotropy_pins","scalar")
        ):
            if kw.get(key) is not None:
                kw[key]=self._pins(kw[key],kind)
        if kw.get("shadow") is not None:
            dx,dy,sigma,strength,*rest=kw["shadow"]
            A=self._linear()
            dv=A@np.array([dx,dy],np.float32)
            sx,sy=self._scale()
            sig=float(sigma)*math.sqrt(max(sx*sy,1e-12))
            kw["shadow"]=(float(dv[0]),float(dv[1]),sig,strength,*rest)
        if kw.get("blur") is not None:
            sx,sy=self._scale()
            kw["blur"]=float(kw["blur"])*math.sqrt(max(sx*sy,1e-12))
        if kw.get("glow") is not None:
            sx,sy=self._scale()
            kw["glow"]=float(kw["glow"])*math.sqrt(max(sx*sy,1e-12))
        return self.c.patch(self.path(boundary),self._pins(color_pins),**kw)

    def stroke(self,points,width,color="#ffffff",alpha=1.0,**kw):
        kw["depth"]=self.depth_offset+float(kw.get("depth",0.0))
        sx,sy=self._scale()
        scale=math.sqrt(max(sx*sy,1e-12))
        if isinstance(width,(tuple,list,np.ndarray)):
            w=tuple(float(x)*scale for x in width)
        else:
            w=float(width)*scale
        if "blur" in kw: kw["blur"]=float(kw["blur"])*scale
        return self.c.stroke(self.path(points),w,color,alpha,**kw)

    def surface(self,boundary,color_pins,**kw):
        kw["depth"]=self.depth_offset+float(kw.get("depth",0.0))
        for key,kind in (
            ("height_pins","scalar"),("normal_pins","normal"),
            ("roughness_pins","scalar"),("reflectance_pins","scalar"),
            ("emission_pins","scalar"),("orientation_pins","orientation"),
            ("anisotropy_pins","scalar")
        ):
            if kw.get(key) is not None:
                kw[key]=self._pins(kw[key],kind)
        surf=self.c.surface(self.path(boundary),self._pins(color_pins),**kw)
        return LocalSurface(self,surf)

    def field(self,rgb=None,coverage=None,depth=0.0,mode="over",emission=None,clip=True):
        depth=self.depth_offset+float(depth)
        inv=np.linalg.inv(self.M)
        ones=np.ones_like(self.c.u)
        X=inv[0,0]*self.c.u+inv[0,1]*self.c.v+inv[0,2]*ones
        Y=inv[1,0]*self.c.u+inv[1,1]*self.c.v+inv[1,2]*ones
        inside=((X>=0)&(X<=1)&(Y>=0)&(Y<=1)).astype(np.float32) if clip else 1.0
        if callable(rgb): col=np.asarray(rgb(X,Y),np.float32)
        else: col=rgb
        if coverage is None: cov=inside
        elif callable(coverage): cov=np.asarray(coverage(X,Y),np.float32)*inside
        else: cov=np.asarray(coverage,np.float32)*inside
        return self.c.field(col,cov,depth=depth,mode=mode,emission=emission)


class LocalSurface:
    """Local-coordinate wrapper around SharedSurface."""
    def __init__(self,group,surface):
        self.g=group
        self.s=surface

    def paint(self,boundary,color_pins=None,**kw):
        for key,kind in (
            ("roughness_pins","scalar"),("reflectance_pins","scalar"),
            ("emission_pins","scalar"),("orientation_pins","orientation"),
            ("anisotropy_pins","scalar"),("normal_pins","normal")
        ):
            if kw.get(key) is not None:
                kw[key]=self.g._pins(kw[key],kind)
        self.s.paint(self.g.path(boundary),self.g._pins(color_pins) if color_pins else None,**kw)
        return self

    def render(self,**kw):
        if kw.get("shadow") is not None:
            dx,dy,sigma,strength,*rest=kw["shadow"]
            A=self.g._linear()
            dv=A@np.array([dx,dy],np.float32)
            sx,sy=self.g._scale()
            kw["shadow"]=(float(dv[0]),float(dv[1]),float(sigma)*math.sqrt(max(sx*sy,1e-12)),strength,*rest)
        if "blur" in kw:
            sx,sy=self.g._scale(); kw["blur"]=float(kw["blur"])*math.sqrt(max(sx*sy,1e-12))
        if "glow" in kw:
            sx,sy=self.g._scale(); kw["glow"]=float(kw["glow"])*math.sqrt(max(sx*sy,1e-12))
        self.s.render(**kw)
        return self

    def oriented_relief(self,orientation_fn,**kw):
        # Convert a local tangent function into global screen angle.
        def fn(u,v):
            inv=np.linalg.inv(self.g.M)
            X=inv[0,0]*u+inv[0,1]*v+inv[0,2]
            Y=inv[1,0]*u+inv[1,1]*v+inv[1,2]
            a=orientation_fn(X,Y)
            A=self.g._linear()
            ca=np.cos(a); sa=np.sin(a)
            vx=A[0,0]*ca+A[0,1]*sa
            vy=A[1,0]*ca+A[1,1]*sa
            return np.arctan2(vy,vx)
        sx,sy=self.g._scale()
        kw["flow_length"]=float(kw.get("flow_length",.020))*math.sqrt(max(sx*sy,1e-12))
        self.s.oriented_relief(fn,**kw)
        return self

    def noise_relief(self,**kw):
        self.s.noise_relief(**kw)
        return self




# ============================================================
# PHASELET2D — localized oriented image-frequency basis
# ============================================================

def _phaselet_atom_params(atom):
    if isinstance(atom,dict):
        x=float(atom['x']); y=float(atom['y'])
        length=float(atom.get('length',atom.get('l',.05)))
        width=float(atom.get('width',atom.get('w',.012)))
        angle=float(atom.get('angle',0.0)); freq=float(atom.get('freq',1.0))
        phase=float(atom.get('phase',0.0)); amp=atom.get('amp',1.0)
    else:
        x,y,length,width,angle,freq,phase,amp=atom
        x=float(x); y=float(y); length=float(length); width=float(width)
        angle=float(angle); freq=float(freq); phase=float(phase)
    return x,y,length,width,angle,freq,phase,amp


def _phaselet_add(canvas,out,atom):
    """Add one atom only inside its ~3.5 sigma support box."""
    x,y,L,W,angle,freq,phase,amp=_phaselet_atom_params(atom)
    L=max(abs(L),1e-5); W=max(abs(W),1e-5)
    ca=math.cos(angle); sa=math.sin(angle)
    # conservative axis-aligned support of rotated anisotropic Gaussian
    k=3.5
    rx=k*(abs(ca)*L+abs(sa)*W)
    ry=k*(abs(sa)*L+abs(ca)*W)
    x0=max(0,int(math.floor((x-rx)*canvas.S)))
    x1=min(canvas.S,int(math.ceil((x+rx)*canvas.S))+1)
    y0=max(0,int(math.floor((y-ry)*canvas.S)))
    y1=min(canvas.S,int(math.ceil((y+ry)*canvas.S))+1)
    if x1<=x0 or y1<=y0:return
    U=canvas.u[y0:y1,x0:x1]; V=canvas.v[y0:y1,x0:x1]
    dx=U-x; dy=V-y
    tang=dx*ca+dy*sa
    nor=-dx*sa+dy*ca
    env=np.exp(-.5*((tang/L)**2+(nor/W)**2)).astype(np.float32)
    wave=np.cos(2*math.pi*freq*(nor/W)+phase).astype(np.float32)
    f=env*wave
    a=np.asarray(amp,np.float32)
    if out.ndim==2:
        if a.ndim==0: out[y0:y1,x0:x1]+=f*float(a)
        else: out[y0:y1,x0:x1]+=f*float(np.dot(a.ravel()[:3],[.2126,.7152,.0722]))
    else:
        if a.ndim==0: out[y0:y1,x0:x1]+=f[...,None]*float(a)
        else: out[y0:y1,x0:x1]+=f[...,None]*a.reshape(1,1,-1)[:,:,:3]


def phaselet_field(canvas, atoms):
    """Sum localized Phaselet2D coefficients efficiently."""
    atoms=list(atoms or [])
    if not atoms:return np.zeros((canvas.S,canvas.S),np.float32)
    amp=_phaselet_atom_params(atoms[0])[-1]
    rgb=np.asarray(amp).ndim>0 and np.asarray(amp).size>1
    out=np.zeros((canvas.S,canvas.S,3),np.float32) if rgb else np.zeros((canvas.S,canvas.S),np.float32)
    for atom in atoms:_phaselet_add(canvas,out,atom)
    return out


def _sample_phaselet_run(points, spacing=.028):
    """Sample a path into centers+tangent angles in the same coordinate space."""
    P=np.asarray(sample_path(points,curve_steps=28) if points and isinstance(points[0][0],str) else points,np.float32)
    if len(P)<2:return []
    seg=np.linalg.norm(P[1:]-P[:-1],axis=1)
    cum=np.r_[0,np.cumsum(seg)]
    total=float(cum[-1])
    if total<1e-8:return []
    n=max(2,int(total/max(float(spacing),1e-4))+1)
    ts=np.linspace(0,total,n)
    centers=[]
    for t in ts:
        j=min(len(seg)-1,max(0,int(np.searchsorted(cum,t,side='right')-1)))
        den=max(float(seg[j]),1e-8)
        a=np.clip((t-cum[j])/den,0,1)
        q=P[j]*(1-a)+P[j+1]*a
        tangent=P[j+1]-P[j]
        ang=math.atan2(float(tangent[1]),float(tangent[0]))
        centers.append((float(q[0]),float(q[1]),ang))
    return centers


def phaselet_run(points, *, spacing=.028, length=.035, width=.008,
                 freq=1.0, phase=0.0, amp=1.0, phase_step=0.0):
    """Compile a spatial path into ordinary phaselet atoms.

    This is syntax compression only: the renderer still receives nothing but
    independent phaselet coefficients.
    """
    centers=_sample_phaselet_run(points,spacing)
    out=[]
    for i,(x,y,a) in enumerate(centers):
        out.append((x,y,float(length),float(width),a,float(freq),float(phase)+i*float(phase_step),amp))
    return out


def _transform_phaselet_atom(group,atom):
    if isinstance(atom,dict):
        x=float(atom['x']); y=float(atom['y']); length=float(atom.get('length',atom.get('l',.05)))
        width=float(atom.get('width',atom.get('w',.012))); angle=float(atom.get('angle',0));
        freq=float(atom.get('freq',1)); phase=float(atom.get('phase',0)); amp=atom.get('amp',1.0)
    else:
        x,y,length,width,angle,freq,phase,amp=atom
    gx,gy=group._xy(x,y)
    A=group._linear()
    t=A@np.array([math.cos(float(angle)),math.sin(float(angle))],np.float32)
    n=A@np.array([-math.sin(float(angle)),math.cos(float(angle))],np.float32)
    lt=max(float(np.linalg.norm(t)),1e-8); ln=max(float(np.linalg.norm(n)),1e-8)
    ga=math.atan2(float(t[1]),float(t[0]))
    return (gx,gy,float(length)*lt,float(width)*ln,ga,float(freq),float(phase),amp)


def _transform_phaselets(group,atoms):
    return [_transform_phaselet_atom(group,a) for a in atoms]


def _phaselet_surface_detail(self, atoms, *, channel='relief', strength=1.0,
                             clamp=.35, smooth=0.0):
    """Apply phaselets *inside one authoritative Alpha surface*.

    channel:
      relief       -> modifies normals through a height-like frequency field
      albedo       -> signed linear RGB modulation of surface color
      emission     -> positive HDR emissive detail
      roughness    -> scalar material modulation
      reflectance  -> scalar material modulation
    """
    atoms=_transform_phaselets(self.g,atoms)
    f=phaselet_field(self.s.c,atoms)
    m=self.s.mask.astype(np.float32)
    if smooth and float(smooth)>0:
        if f.ndim==2:f=gaussian_filter(f,max(.3,float(smooth)*self.s.c.S))
        else:f=gaussian_filter(f,(max(.3,float(smooth)*self.s.c.S),max(.3,float(smooth)*self.s.c.S),0))
    st=float(strength)
    if channel=='relief':
        if f.ndim==3:f=f[...,0]*.2126+f[...,1]*.7152+f[...,2]*.0722
        self.s.relief_field(f, strength=st, mask=m)
    elif channel=='albedo':
        if f.ndim==2:f=f[...,None]
        self.s.color=np.clip(self.s.color + f*st*m[...,None],0,1)
    elif channel=='emission':
        if f.ndim==2:f=f[...,None]
        self.s.emit=np.maximum(0,self.s.emit + np.maximum(f,0)*st*m[...,None])
    elif channel=='roughness':
        if f.ndim==3:f=f[...,0]*.2126+f[...,1]*.7152+f[...,2]*.0722
        self.s.rough=np.clip(self.s.rough+f*st*m,.02,1)
    elif channel=='reflectance':
        if f.ndim==3:f=f[...,0]*.2126+f[...,1]*.7152+f[...,2]*.0722
        self.s.refl=np.clip(self.s.refl+f*st*m,0,1)
    else:
        raise ValueError("phaselet channel must be relief/albedo/emission/roughness/reflectance")
    return self


def _phaselet_surface_run(self, points, *, channel='relief', spacing=.028,
                          length=.035,width=.008,freq=1.0,phase=0.0,amp=1.0,
                          phase_step=0.0,strength=1.0,clamp=.35,smooth=0.0):
    atoms=phaselet_run(points,spacing=spacing,length=length,width=width,freq=freq,
                       phase=phase,amp=amp,phase_step=phase_step)
    return _phaselet_surface_detail(self,atoms,channel=channel,strength=strength,
                                    clamp=clamp,smooth=smooth)

LocalSurface.phaselets=_phaselet_surface_detail
LocalSurface.phaselet_run=_phaselet_surface_run


def _phaselet_canvas_layer(self, atoms, *, coverage=None, depth=0.0,
                           mode='add', strength=1.0, glow=0.0):
    f=phaselet_field(self,atoms)
    if f.ndim==2:f=f[...,None]
    cov=np.ones((self.S,self.S),np.float32) if coverage is None else np.asarray(coverage,np.float32)
    rgb=f*float(strength)
    if mode=='emission':
        pos=np.maximum(rgb,0)
        self.field(pos,cov,depth=depth,mode='add')
        if glow and float(glow)>0:
            self.hdr += gaussian_filter(pos*cov[...,None],(float(glow)*self.S,float(glow)*self.S,0))*.70
    else:
        self.field(rgb,cov,depth=depth,mode=mode)
    return self

AlphaCanvas.phaselets=_phaselet_canvas_layer


def _local_phaselet_layer(self, atoms, **kw):
    atoms=_transform_phaselets(self,atoms)
    kw['depth']=self.depth_offset+float(kw.get('depth',0.0))
    return self.c.phaselets(atoms,**kw)


def _local_phaselet_run(self, points, **kw):
    atoms=phaselet_run(points,
        spacing=kw.pop('spacing',.028), length=kw.pop('length',.035),
        width=kw.pop('width',.008), freq=kw.pop('freq',1.0),
        phase=kw.pop('phase',0.0), amp=kw.pop('amp',1.0),
        phase_step=kw.pop('phase_step',0.0))
    return _local_phaselet_layer(self,atoms,**kw)

LocalGroup.phaselets=_local_phaselet_layer
LocalGroup.phaselet_run=_local_phaselet_run




def _phaselet_surface_texture(self, *, seed=1, count=180, center=(.5,.5), radius=(.5,.5),
                              angle=0.0, angle_jitter=.35,
                              length=(.012,.040), width=(.0025,.009), freq=(.7,2.4),
                              channel='relief', strength=.02, tint=(1,1,1),
                              phase_random=True, radial_angle=False, highpass=0.0, compress=2.4):
    """Dense Gabor/phaselet packet texture inside an authoritative surface.

    This is a compression syntax for many ordinary Phaselet2D coefficients.
    It does not introduce semantic texture types.  The LLM controls scale,
    anisotropy, orientation, frequency and amplitude directly.
    """
    rng=np.random.default_rng(int(seed))
    cx,cy=float(center[0]),float(center[1])
    rx,ry=float(radius[0]),float(radius[1])
    l0,l1=(float(length[0]),float(length[1])) if isinstance(length,(tuple,list)) else (float(length),float(length))
    w0,w1=(float(width[0]),float(width[1])) if isinstance(width,(tuple,list)) else (float(width),float(width))
    f0,f1=(float(freq[0]),float(freq[1])) if isinstance(freq,(tuple,list)) else (float(freq),float(freq))
    atoms=[]
    for _ in range(int(count)):
        # Uniform-ish sampling in an authored ellipse.
        rr=math.sqrt(rng.random()); th=rng.random()*math.tau
        x=cx+math.cos(th)*rx*rr; y=cy+math.sin(th)*ry*rr
        L=math.exp(rng.uniform(math.log(max(l0,1e-5)),math.log(max(l1,1e-5))))
        W=math.exp(rng.uniform(math.log(max(w0,1e-5)),math.log(max(w1,1e-5))))
        fr=rng.uniform(f0,f1)
        if radial_angle:
            aa=math.atan2(y-cy,x-cx)+math.pi/2
        else:
            aa=float(angle)
        aa += rng.normal(0,float(angle_jitter))
        ph=rng.random()*math.tau if phase_random else 0.0
        atoms.append((x,y,L,W,aa,fr,ph,1.0))

    atoms=_transform_phaselets(self.g,atoms)
    field=phaselet_field(self.s.c,atoms)
    m=self.s.mask.astype(np.float32)
    vals=field[m>.12]
    if vals.size:
        field=(field-float(vals.mean()))/max(float(vals.std()),1e-6)
    if highpass and float(highpass)>0:
        field=field-gaussian_filter(field,max(.35,float(highpass)*self.s.c.S))
        vals=field[m>.12]
        if vals.size: field=field/max(float(vals.std()),1e-6)
    if compress and float(compress)>0:
        cc=float(compress); field=np.tanh(field/cc)*cc
    st=float(strength)
    if channel=='relief':
        self.s.relief_field(field,strength=st,mask=m)
    elif channel=='albedo':
        t=np.asarray(tint,np.float32)
        self.s.color=np.clip(self.s.color+field[...,None]*t.reshape(1,1,3)*st*m[...,None],0,1)
    elif channel=='roughness':
        self.s.rough=np.clip(self.s.rough+field*st*m,.02,1)
    elif channel=='reflectance':
        self.s.refl=np.clip(self.s.refl+field*st*m,0,1)
    elif channel=='emission':
        t=np.asarray(tint,np.float32)
        self.s.emit=np.maximum(0,self.s.emit+np.maximum(field,0)[...,None]*t.reshape(1,1,3)*st*m[...,None])
    else:
        raise ValueError('unknown phaselet texture channel')
    return self

LocalSurface.phaselet_texture=_phaselet_surface_texture


def _root_group(self):
    return LocalGroup(self)

AlphaCanvas.root_group=_root_group


# -----------------------------------------------------------------------------
# PERFORMANCE FIX: draw ordinary patches in their own pixel bounding box.
# AlphaCanvas.patch originally allocated and shaded S×S arrays for every patch,
# even 1-pixel dust motes. This is geometrically equivalent for ordinary opaque
# patches; complex glass/shadow/blur/height effects retain the proven old path.
# -----------------------------------------------------------------------------
_ALPHA_ORIGINAL_PATCH = AlphaCanvas.patch

def _alpha_fast_local_patch(self, boundary, color_pins, *, height_pins=None,
                            normal_pins=None, roughness_pins=None,
                            reflectance_pins=None, emission_pins=None,
                            orientation_pins=None, anisotropy_pins=None,
                            color_continuity=1, height_continuity=1,
                            alpha=1.0, smooth=False, depth=0.0,
                            transmission=0.0, ior=1.45, absorption=None,
                            thickness=.5, unlit=False, lighting_mix=1.0,
                            glow=0.0, blur=0.0, shadow=None, blend="normal",
                            background_sample_shift=(0,0)):
    # Keep original handling for effects that depend on surrounding pixels.
    if (height_pins or float(transmission)>0 or float(glow)>0 or
        float(blur)>0 or shadow is not None or blend!="normal" or
        (background_sample_shift != (0,0))):
        return _ALPHA_ORIGINAL_PATCH(
            self,boundary,color_pins,height_pins=height_pins,
            normal_pins=normal_pins,roughness_pins=roughness_pins,
            reflectance_pins=reflectance_pins,emission_pins=emission_pins,
            orientation_pins=orientation_pins,anisotropy_pins=anisotropy_pins,
            color_continuity=color_continuity,height_continuity=height_continuity,
            alpha=alpha,smooth=smooth,depth=depth,transmission=transmission,
            ior=ior,absorption=absorption,thickness=thickness,unlit=unlit,
            lighting_mix=lighting_mix,glow=glow,blur=blur,shadow=shadow,blend=blend,
            background_sample_shift=background_sample_shift)

    pts=sample_path(boundary) if boundary and isinstance(boundary[0][0],str) else boundary
    pts=np.asarray(pts,np.float32)
    if len(pts)<3:return
    if smooth and len(pts)>=4:
        for _ in range(3):
            qroll=np.roll(pts,-1,axis=0)
            pts=np.stack([.75*pts+.25*qroll,.25*pts+.75*qroll],axis=1).reshape(-1,2)

    S=self.S
    x0=max(0,int(math.floor(float(pts[:,0].min())*S))-3)
    x1=min(S,int(math.ceil(float(pts[:,0].max())*S))+4)
    y0=max(0,int(math.floor(float(pts[:,1].min())*S))-3)
    y1=min(S,int(math.ceil(float(pts[:,1].max())*S))+4)
    if x0>=x1 or y0>=y1:return
    W=x1-x0;H=y1-y0
    im=Image.new("L",(W,H),0)
    ImageDraw.Draw(im).polygon([(float(x)*S-x0,float(y)*S-y0) for x,y in pts],fill=255)
    mask=np.asarray(im,np.float32)/255.0*float(alpha)
    yy,xx=np.where(mask>1e-5)
    if not len(xx):return
    q=np.c_[((xx+x0)+.5)/S,((yy+y0)+.5)/S]
    bounds=(float(xx.min()+x0+.5)/S,float(yy.min()+y0+.5)/S,
            float(xx.max()+x0+.5)/S,float(yy.max()+y0+.5)/S)

    if int(color_continuity)==0:
        C=LocalTriField(color_pins,3,linear_color=True,bounds=bounds)(q).reshape(-1,3)
    else:
        C=SparseField(color_pins,3,linear_color=True)(q).reshape(-1,3)
    C=np.clip(C,0,1)

    N=np.zeros((H,W,3),np.float32)
    N[...,2]=1
    if normal_pins:
        nv=SparseField(normal_pins,2)(q)
        N[yy,xx,0]+=nv[:,0]; N[yy,xx,1]+=nv[:,1]
        xy=np.clip(N[yy,xx,:2],-.98,.98)
        N[yy,xx,2]=np.sqrt(np.maximum(.02,1-(xy*xy).sum(1)))
    N=normalize(N)

    def scalar_field(pins,default):
        if not pins:return np.full(len(q),default,np.float32)
        return SparseField(pins,1)(q)[:,0]

    rough=np.clip(scalar_field(roughness_pins,.45),.02,1)
    refl=np.clip(scalar_field(reflectance_pins,.05),0,1)
    E=SparseField(emission_pins,3,linear_color=False)(q) if emission_pins else np.zeros((len(q),3),np.float32)

    color_img=np.zeros((H,W,3),np.float32)
    rough_img=np.ones((H,W),np.float32)*.45
    refl_img=np.zeros((H,W),np.float32)
    emit_img=np.zeros((H,W,3),np.float32)
    color_img[yy,xx]=C;rough_img[yy,xx]=rough;refl_img[yy,xx]=refl;emit_img[yy,xx]=E

    ori_img=None; aniso_img=None
    if orientation_pins:
        ori_img=np.zeros((H,W),np.float32)
        ori_img[yy,xx]=OrientationField(orientation_pins)(q)[:,0]
    if anisotropy_pins:
        aniso_img=np.zeros((H,W),np.float32)
        aniso_img[yy,xx]=np.clip(scalar_field(anisotropy_pins,0),0,1)
    base_unlit=color_img+emit_img
    if unlit:
        shaded=base_unlit
    else:
        lit=self._shade(color_img,N,rough_img,refl_img,np.zeros_like(emit_img),
                        orientation=ori_img,anisotropy=aniso_img)+emit_img
        lm=float(np.clip(lighting_mix,0,1))
        shaded=base_unlit*(1-lm)+lit*lm

    hdr=self.hdr[y0:y1,x0:x1]
    dview=self.depth[y0:y1,x0:x1]
    a=np.where((mask>1e-5)&(float(depth)>=dview),mask,0).astype(np.float32)
    hdr[:]=hdr*(1-a[...,None])+shaded*a[...,None]
    av=self.alpha[y0:y1,x0:x1]
    av[:]=av+a*(1-av)
    dview[:]=np.where(a>1e-5,depth,dview)
    self.stats['patches']+=1

AlphaCanvas.patch=_alpha_fast_local_patch

def render_art_code(code, output_path, size=512, supersample=2, seed=4):
    """Execute trusted Alpha art payload. Payload must define build(c)."""
    ns = dict(
        np=np, math=math,
        AlphaCanvas=AlphaCanvas,
        SparseField=SparseField,
        SmoothScalarField=SmoothScalarField,
        LocalTriField=LocalTriField,
        OrientationField=OrientationField,
        SharedSurface=SharedSurface,
        LocalGroup=LocalGroup,
        LocalSurface=LocalSurface,
        CompoundBuilder=CompoundBuilder,
        CompoundSharedSurface=CompoundSharedSurface,
        rect=rect,
        ellipse_poly=ellipse_poly,
        mask_from_poly=mask_from_poly,
        sample_path=sample_path,
        tapered_ribbon=tapered_ribbon,
        phaselet_field=phaselet_field,
        phaselet_run=phaselet_run,
        hexrgb=hexrgb,
        srgb_to_linear=srgb_to_linear,
        linear_to_srgb=linear_to_srgb,
        gaussian_filter=gaussian_filter,
        distance_transform_edt=distance_transform_edt,
    )
    ns["__name__"] = "__alpha_art__"
    exec(compile(code, "<ALPHA_ART_CODE>", "exec"), ns, ns)
    if not callable(ns.get("build")):
        raise ValueError("Alpha payload must define build(c)")
    c = AlphaCanvas(size=size, supersample=supersample, seed=seed)
    ns["build"](c)
    finish = ns.get("FINISH", {})
    image = c.finish(**finish)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    meta = {
        "renderer": "Model Alpha V-Phaselet X1.2",
        "size": size,
        "supersample": supersample,
        "seed": seed,
        **c.stats,
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return image, meta


def render_art_file(art_path, output_path, size=512, supersample=2, seed=4):
    return render_art_code(Path(art_path).read_text(), output_path, size, supersample, seed)

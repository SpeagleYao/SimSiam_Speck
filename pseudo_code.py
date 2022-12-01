### Pseudo Code ### 

# h: prediction mlp

def normalize():
    pass
def enc():
    pass
def update():
    pass
def f(): # f: backbone + projection mlp 
    pass
def h(): # predictions, n-by-d 
    pass

loader = None
delta = 0x0040, 0x0000

def D(p, z): # negative cosine similarity 
    z = z.detach() # stop gradient 
    p = normalize(p, dim=1) # l2-normalize 
    z = normalize(z, dim=1) # l2-normalize 
    return -(p*z).sum(dim=1).mean()

for x in loader: # load a minibatch x with n samples 
    c1, c2 = enc(x), enc(x+delta) # random augmentation 
    z1, z2 = f(c1), f(c2) # projections, n-by-d 
    p1, p2 = h(z1), h(z2) # predictions, n-by-d 
    L = D(p1, z2)/2 + D(p2, z1)/2 # loss 
    L.backward() # back-propagate 
    update(f, h) # SGD update 
